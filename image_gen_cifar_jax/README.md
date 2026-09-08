# image_gen_cifar_jax -- session notes

Working notes and results from the JAX/Equinox port + debugging session for the three
CIFAR-10 hierarchical latent-AR image-gen variants: `run_fullattn_noquant.py`,
`run_causalattn.py`, `run_ar_clockwork.py` (`_v1.py` siblings are the original plain-dict-
pytree JAX versions, kept for reference/import -- see each Equinox file's own docstring).

## What was done

- Ported all three to Equinox (`eqx.Module` classes instead of manually-threaded dict
  pytrees), validated against the `_v1` originals via bit-exact primitive checks and
  KV-cache-vs-training-path consistency checks (float32-noise-level agreement, 1e-6 to 1e-7).
- Added checkpoint save/resume (`eqx_common.save_checkpoint`/`load_checkpoint`) to all three
  -- did not exist before this session.
- Added a modular RGB-head system to `run_ar_clockwork.py`: `ParallelRGBHead` (original,
  independent linear heads), `SequentialRGBHead` (DeepSeek-MTP-style causal R->G->B chain via
  real byte embeddings), `DiffusionRGBHead` (discrete-diffusion-style: each channel masked
  independently at `mask_prob`, bidirectional 3-token block predicts the masked ones; default
  single-shot generation starts fully masked, behaving like the parallel head at inference).
- Added a `train_subset_n` overfit-sanity-check mode + `save_compare_grid` (generated vs.
  ground truth, side by side) to all three, specifically to answer "is generation actually
  broken, or just undertrained" -- see Key Finding #1 below.
- Added `eqx_common.sinkgd` -- a drop-in stateless optimizer from arXiv:2502.06742
  ("Gradient Multi-Normalization for Stateless and Scalable LLM Training"): Sinkhorn row/
  column-normalized gradient updates for 2D linear-layer weight matrices (no momentum/
  variance state -- saves ~2x the memory AdamW spends on those params), plain AdamW for
  everything else (norms, embedding tables). Paired with `warmup_const_schedule` (warmup then
  flat, no decay -- SinkGD's stateless-ness makes the decay phase less necessary). All three
  `run_*.py` now accept `--optimizer {adamw,sinkgd}` and `--optimizer_kwargs <dict>`.
  **Implemented and unit-tested locally; not yet deployed to any real training run.**
- Added a warmup -> flat -> cosine-decay LR schedule (`warmup_const_decay_schedule`) to
  `run_ar_clockwork.py` as an alternative to the original flat-forever-after-warmup schedule.

## Key finding #1: the "broken generation" was exposure bias + sampling noise, not a bug

Early runs showed generated samples as pure vertical-scanline noise (fullattn_noquant/
causalattn) or "confetti" speckle (clockwork), no matter how long training ran. Root-caused
via a purpose-built overfit sanity check (`train_subset_n=1000`, `ar_clockwork_overfit1000_*`
configs):

1. **Consistency check** (generation-path vs. training-path h_out, both given identical real
   ground-truth rows): matched to float32 noise (`max_diff ~1e-6`) on a fresh, untrained model
   for all three architectures. This proves `generate()`'s KV-cache stepping logic is a
   faithful re-implementation of the training-time computation -- not a bug.
2. **The actual cause**: at 96.9% teacher-forced train accuracy (clockwork-parallel,
   overfit-1000, epoch 1710), **greedy decoding reproduced the memorized training image
   100% pixel-exact**, while **categorical sampling at temperature=1.0 still produced pure
   noise** on the same checkpoint. With ~96 pixels/row x 31 free-running rows, even a highly
   confident model gets visibly corrupted by compounding independent sampling draws --
   classic exposure bias (the model is only ever trained conditioned on real previous rows,
   never its own generated output).
3. Confirmed the pattern on causalattn's stale codebook utilization too: `col_group_size=1`
   (SISO -- `col_mix` a no-op) gave the encoder's PQ codebooks a util of only ~0.10-0.17
   (should be ~1.0 for full utilization). Switching to `col_group_size=32` (MIMO, full
   cross-column mixing) raised utilization to ~0.56-0.62 in the follow-up overfit-1000 run --
   a real, confirmed improvement, not just theory.

**Practical implication**: `qual_gen_greedy=False, qual_gen_temperature=1.0` (the earlier
default) is close to useless as a training-progress diagnostic until a model is very nearly
converged. Configs now default to `qual_gen_temperature=0.05`.

## Key finding #2: architecture speed ranking differs between tiny-overfit and full-dataset

| Setting | Fastest to overfit | Notes |
|---|---|---|
| `train_subset_n=1000` (tiny, easy to memorize) | clockwork-**parallel** (96.9% @ ep.1710) > clockwork-sequential (68.8% after the param-parity fix, was stuck ~15% before) >> fullattn_noquant (51-60%) >> causalattn (worst, ~17-30%, codebook-collapse-limited) | |
| Full 50k train split, 3000 epochs, LR warmup->flat->cosine-decay | clockwork-**sequential** > clockwork-**diffusion** | at matched epoch counts (e.g. ep.850-900) sequential's train bpb_main ~4.8-4.9 vs. diffusion's ~5.4-5.6; diffusion is notably the slowest to converge of the two tested on full data |

Root cause for clockwork-sequential's initial tiny-overfit stall: **not a bug** -- the
sequential (MTP-style) head originally had only ~96K params vs. the parallel head's ~6.29M
(a 65x capacity gap), simply starved. Fixed by scaling `mtp_dim` 64->640 for param parity
(~6.9M); confirmed the fix worked (68.8% vs ~15-17% at matched epoch count).

The auxiliary NTP head (single-next-pixel anchor task, dropped at generation time) overfits
far harder than the main task on both clockwork variants with no generalization at all: train
`acc_ntp` reaches 0.86-0.94 while val `acc_ntp` stays pinned near-random (~0.038) and val
`bpb_ntp` blows up to 46-65 (vs. train ~0.4-0.8) -- expected for an unregularized anchor loss,
not itself a problem since it's never used for sampling.

## Full-dataset training results so far (500k-image full CIFAR-10, in progress)

Config: 6-level rise-then-fall sandwich strides `(1,2,4,4,2,1)`, ~27M params, `lr=5e-4`,
1 epoch warmup -> 10 epochs flat -> cosine decay over the rest, `weight_decay=1e-5`,
`epochs=3000`. Explicitly optimizing to overfit the full train set as hard as possible --
val is not the target and is expected (and observed) to keep worsening.

**tpu3 (clockwork-diffusion)**

| Epoch | Train bpb_main | Val bpb_main | Train acc_main | Val acc_main | Train bpb_ntp | Val bpb_ntp | Train acc_ntp | Val acc_ntp |
|---|---|---|---|---|---|---|---|---|
| 400 | 5.55 | 7.47 | 0.106 | 0.053 | 0.636 | 46.2 | 0.903 | 0.039 |
| 500 | 5.53 | 7.57 | 0.097 | 0.053 | 0.663 | 48.4 | 0.891 | 0.039 |
| 600 | 5.45 | 7.73 | 0.107 | 0.053 | 0.720 | 54.6 | 0.874 | 0.038 |
| 700 | 5.58 | 7.91 | 0.092 | 0.052 | 0.504 | 56.2 | 0.922 | 0.038 |
| 800 | 5.38 | 8.15 | 0.111 | 0.052 | 0.402 | 61.0 | 0.943 | 0.038 |
| 850 | 5.42 | 8.19 | 0.108 | 0.052 | 0.563 | 63.0 | 0.905 | 0.038 |

**tpu4 (clockwork-sequential)**

| Epoch | Train bpb_main | Val bpb_main | Train acc_main | Val acc_main | Train bpb_ntp | Val bpb_ntp | Train acc_ntp | Val acc_ntp |
|---|---|---|---|---|---|---|---|---|
| 450 | 4.72 | 6.43 | 0.158 | 0.083 | 0.825 | 48.7 | 0.861 | 0.039 |
| 550 | 4.90 | 6.59 | 0.129 | 0.083 | 0.650 | 54.8 | 0.896 | 0.039 |
| 650 | 4.84 | 6.64 | 0.133 | 0.083 | 0.768 | 54.5 | 0.867 | 0.039 |
| 750 | 4.87 | 6.75 | 0.132 | 0.083 | 0.674 | 59.1 | 0.885 | 0.038 |
| 850 | 4.77 | 6.92 | 0.148 | 0.082 | 0.466 | 64.5 | 0.931 | 0.038 |
| 900 | 4.88 | 6.92 | 0.133 | 0.082 | 0.562 | 64.6 | 0.907 | 0.038 |

Both train bpb_main curves are plateauing/oscillating (not monotonically dropping) rather
than continuing to improve -- neither has converged. sequential remains consistently ahead of
diffusion at matched epochs.

**tpu1/tpu2** (`train_subset_n=1000`, MIMO `col_group_size=32`, 5000 epochs, both finished):
fullattn_noquant reached 72-78% train acc (val bpb blown up to ~34.7, as expected for this
deliberate overfit test); causalattn reached ~28-30% train acc with healthy codebook
utilization (~0.57-0.62, val bpb ~5.8) -- MIMO fixed the codebook-collapse problem but
causalattn is still the slowest of the four to memorize even a 1000-image set.

## KIV (keep-in-view) -- too hard / deprioritized for now

- **Diffusion head tuning**: implemented and functionally verified, but is the slowest
  architecture to converge on both the overfit-1000 and full-dataset tests. True iterative
  multi-step remasking/refinement at generation time (beyond the single-shot fully-masked
  default) was explicitly not implemented -- would need real diffusion-style sampling to be a
  fair comparison, and isn't yet worth the effort given it's already behind on the single-shot
  metric. Parked until the higher-priority architecture work below is settled.
- **SinkGD optimizer**: implemented, unit-tested locally, not yet tried on any real training
  run. Parked pending a decision on which run to try it on first.
- **col_group_size sweep beyond SISO/MIMO** ("grouped", an intermediate value): not explored
  -- only the two extremes were tested.

## Way forward (proposed, NOT yet implemented -- pending confirmation)

Plan discussed for evolving `run_causalattn.py` toward the original `qcute_lagcodec` design
more faithfully (tentative rename to `run_causalcodec.py`/"causalcodec"):

- Change the NTP auxiliary target: currently it predicts the *next row* (a forward/lookahead
  target). The proposal is to make each of the 32 scanlines predict *only that scanline*
  (no forward-looking target at all) -- i.e. drop the "predict ahead" framing entirely for the
  per-scanline heads.
- Under this change, genuine look-ahead prediction (if wanted at all) would only be possible
  via the encoder LM side, and a true bits-per-byte figure may only be computable from the
  encoder's level-0 output specifically, not the full hierarchy -- flagged as uncertain by design
  intent, needs to be worked out concretely before implementing.
- Status: this was being restated for confirmation when interrupted -- **not yet agreed or
  implemented**. Next step is to re-confirm the exact scoping (what "no forward" means
  precisely for training loss vs. bpb reporting) before touching code.
