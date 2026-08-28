# summformer_jax

JAX/Flax NNX port of `qcute/summformer/summformer.py`, made GPT2-like (LayerNorm, plain GELU MLP,
plain MHA -- no RMSNorm/SwiGLU/GQA/QK-norm from the PyTorch original) and extended to 3
`pos_method`s matching `gpt2_jax`'s convention (`rope`/`learnable`/`base`). Runs an **ablation of
the Ks-hierarchical-summarization + fuse-cross-attn method against [../gpt2_jax](../gpt2_jax)'s
plain-GPT2 baselines**, at matched dataset/vocab/optimizer/schedule/`total_batch_size` -- the goal
is an honest comparison at roughly matched compute, not tuning to win. Full background, the
FLOPs/param derivation behind the current hparams, and the running log of what's been tried:
[../docs/status_tpu.md](../docs/status_tpu.md).

This doc is the step-by-step for a fresh session picking this lineage back up on a TPU node. Env
setup, data prep, `tpu-info` monitoring, and the checkpoint-egress policy are IDENTICAL to
`gpt2_jax`'s own README (same node, same `~/qcute` checkout, same `data/fineweb-edu-10B` dataset
-- see [../gpt2_jax/README.md](../gpt2_jax/README.md) for those, not repeated here).
**Never create a TPU yourself** — only use nodes already listed in [../TPU.md](../TPU.md).

## Run

```bash
tmux new-session -d -s <run_name> "\
  export LD_LIBRARY_PATH=/home/muaz/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/lib:\$LD_LIBRARY_PATH && \
  export PYTHONPATH=summformer_jax && \
  cd ~/qcute && source .venv/bin/activate && \
  python3 summformer_jax/train_summformer.py --config configs/summformer_jax/medium_rope_ablation.py --run-name <run_name> \
    > ~/<run_name>.log 2>&1; echo TRAIN_EXIT=\$? >> ~/<run_name>.log"
```

`--config configs/summformer_jax/medium_rope_ablation.py` (vs. `gpt2_jax/medium_rope_default.py`)
or `small_rope_ablation.py` (vs. `small_rope_default.py`) -- same `--config` convention as
`gpt2_jax`, CLI flags override. Dataset is the SAME `data/fineweb-edu-10B` GPT2-BPE token shards
`gpt2_jax` uses (`Config.vocab_size=50304` by default, not the byte alphabet -- byte-vocab mode
still exists via `--vocab-size 0`/`summformer_jax/dataset_preparation.py`, see model docstring,
but isn't what these ablation configs use).

## How `_cascade` works (worked example, `Ks=(2,2,2)`, `L=8`)

`n_fuse = len(Ks) = 3`, so 4 levels exist (`lms[0..3]`) and 3 fuse stages (`fuse_stages[0..2]`).

**Level-0 pass**: `h = _run_blocks(0, embed(token_ids), ...)` -- ordinary causal self-attention over
the raw 8 tokens. `cur_h = x_cross = h`, both `(B, 8, D)`.

**Stage `s`, loop over `range(n_fuse)`** -- `cur_h` is the *previous* stage's compressed code
sequence (or the level-0 output for `s=0`); `x_cross` is always the full token-length stream:

| Stage | `cur_h` len (in) | `K_s` | KV len out (`h_code`) | `cum_K` | `code_pos_abs` | `x_cross` len (Q, unchanged) |
|---|---|---|---|---|---|---|
| level-0 | -- | -- | -- | -- | -- | **8** |
| `s=0` | 8 | 2 | **4** | 2 | `[1,3,5,7]` | 8 |
| `s=1` | 4 | 2 | **2** | 4 | `[3,7]` | 8 |
| `s=2` | 2 | 2 | **1** | 8 | `[7]` | 8 |

Per stage:
1. `code_h = cur_h[:, K_s-1::K_s, :][:, :n_blocks, :]` -- picks the *last* token's hidden state of
   each non-overlapping `K_s`-sized block (no learned pooling, just strided selection).
2. `h_code = _run_blocks(s+1, code_h, ...)` -- a **separate** transformer stack (`lms[s+1]`, own
   weights) runs causal self-attention over just this stage's compressed code sequence.
3. `code_pos_abs = (arange(n_blocks)+1)*cum_K - 1` -- maps each code back to the absolute raw-token
   position it represents (needed for both RoPE and causal gating).
4. `fuse_mask`: query position `q` may only attend to a code whose `code_pos_abs <= q` -- e.g. at
   `s=2` only position 7 ever sees the single global summary (`code_pos_abs=[7]`); everyone else's
   row is all-masked, which `sdpa_with_sink` resolves to attending only the zero sink (a no-op for
   them this stage). Strict causality: you can never see a summary of a block containing future
   tokens.
5. `x_cross = fuse_stages[s](x_cross, h_code, ..., fuse_mask, ...)` -- cross-attention (all 8 query
   positions against this stage's `h_code` KV) + MLP refinement (`FuseStage`'s own weights).
6. `x_cross = _run_blocks(0, x_cross, ...)` -- **reruns `lms[0]`** (same weights as the initial
   pass -- "shared across token-level pass and refinement") so tokens re-mix after absorbing the
   injected summary.
7. `cur_h = h_code` -- the next stage cascades on *this* stage's compressed sequence.

**Shape summary**: `x_cross` (the query stream) is `(B, L, D)` for the whole cascade -- only its
*content* is progressively updated. What shrinks is the KV side each stage cross-attends to:
`KV_len(s) = L // prod(Ks[0..s])` (`8 → 4 → 2 → 1` above). So each cross-attention call costs
`O(L * L/cum_K)`, not `O(L^2)`, geometrically cheaper as `cum_K` grows -- at the real 1024-context
scale, stage 0's KV is 512, stage 1's is 256, stage 2's is 128, vs. a dense decoder's every-layer
`L^2`. Measured end-to-end (`scripts/compare_summformer_gpt2.py`, medium ablation vs. medium
baseline, both rope, matched `d_model`/`n_heads`/context): **0.65x the FLOPs, 3.0x smaller
single-sequence KV cache** (`(1+n_fuse)*n_layers=8` effective cached layers vs. gpt2-medium's 24 --
current ablation configs use `attn_window=None`/unbounded, so this saving is purely from fewer
cached layers, not from any KV sequence-length compression).

## What works (confirmed 2026-08-27)

- **Model correctness**: forward pass correct for all 3 `pos_method`s (loss ≈ ln(vocab) at random
  init), backward pass clean (no NaNs), and **KV-cache consistency confirmed bit-exact
  (`match_rate=1.0`) for all 3 `pos_method`s** against the full-recompute reference
  (`check_kv_cache_consistency`) -- both at byte-vocab and BPE-vocab scale.
- **Dataloader audit**: `data_loader.py` is functionally byte-identical to `gpt2_jax/data_loader.py`
  (confirmed via diff, only the module docstring differs), which is itself an algorithm-identical
  port of Cable's own `data_loader.py` -- so the ablation and its baseline share exactly the same
  data pipeline, not just the same dataset files.
- **Both ablation sizes compile and train without OOM** at the hparams below, `total_batch_size=524288`
  matched to the paper (see `gpt2_jax/README.md`'s formula section) with a conservative per-device
  `batch_size=4` (this architecture's fuse-stage compute is heavier per forward pass than a plain
  GPT2 block, so a smaller per-device batch was the starting choice -- not yet tuned upward).

## Formula to match baselines (params/FLOPs, not just total_batch_size)

Each fuse stage costs ~`fuse_n_layers + n_layers` extra full-byte-length-equivalent layers (the
cross-attn `FuseStage` and the post-cross-attn refinement pass both run at full sequence length,
not the short pooled length) on top of the initial `n_layers` byte pass. So:

```
effective_depth ≈ n_layers * (1 + 2 * n_fuse)          # n_fuse = len(Ks) -- no trailing dummy entry
params ≈ effective_depth * 12 * d_model^2 + 2 * vocab_size * d_model   # untied head
```

(As of 2026-08-27, `Ks`'s length equals `n_fuse` exactly -- no top-level placeholder entry. Older
notes/configs referencing `Ks=(2,2,2,2)` predate this fix; the equivalent config today is
`Ks=(2,2,2)`, same `n_fuse=3`, same architecture -- see status_tpu.md's "Ks tuple semantics fixed"
note.)

To land near a GPT2 baseline's own `(n_layer, d_model)` at a given `Ks`, solve for `n_layers`
given `effective_depth ≈ baseline_n_layer`. Current hparams (both use `Ks=(2,2,2)`, `n_fuse=3`,
chosen after comparing several `Ks`-length/n_layers tradeoffs -- see status_tpu.md for the
derivation and the numbers for other `Ks` lengths that were considered and rejected):

| Ablation | vs. baseline | d_model | n_heads | n_layers | effective_depth | params | baseline params | delta |
|---|---|---|---|---|---|---|---|---|
| medium | gpt2-medium (24L/1024d) | 1024 | 16 | 2 | 14 | ~279.3M | 353.8M | -21% |
| small | gpt2-small (12L/768d) | 768 | 12 | 1 | 7 | ~126.8M | ~123.6M | +2.6% |

## Status (2026-08-27)

| Node | Run | tok/s | Notes |
|---|---|---|---|
| tpu5 | `summformer_medium_ablation` | ~250.2K | vs. tpu4's `medium_paper_match_b8` (~102.6K) |
| tpu7 | `summformer_small_ablation` | ~599K | vs. tpu6's `small_paper_match` (~257.8K) |

Both ablation runs are currently *faster* in raw tok/s than their GPT2 baselines despite the extra
fuse-stage compute -- attributable to `n_layers` being much smaller (2 and 1) than the baselines'
(24 and 12) even after the effective-depth multiplier, not evidence the architecture is cheaper
per-effective-layer. Loss/bpb comparisons (the actual point of the ablation) are what matters, not
raw throughput -- see status_tpu.md once both runs have logged enough steps to compare bpb at
matched token counts.
