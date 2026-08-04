# Status

Tracks progress against the phase plan in
[continuous_tokenizer_handover.md §5](continuous_tokenizer_handover.md#5-implementation-phases).

## Phase 0 — infrastructure

- [x] Byte-level data pipeline (`datasets/enwik8.gz` + `datasets/enwik8_1M.gz`
      1,000,000-byte subset via `scripts/prepare_data.py`).
- [x] Reference baseline: byte-level MTP LM (`qcute/bytelm.py`, `xs` preset
      ≈3.7M params, bandwidth-matched to `qcute.qcutelm`'s K=8 via 8 MTP heads),
      reports exact BPB.
- [x] BPE baseline (`qcute/bpelm.py` + `scripts/train_bpe.py`, sentencepiece,
      no MTP — bandwidth from BPE merging alone). vocab=8192 (power-of-2)
      reaches ~3.3 bytes/token on the tiny corpus. Tokenizer trained with
      `identity` normalization / no whitespace collapsing / byte_fallback /
      no dummy prefix, so encode→decode is a verified lossless bijection —
      `bits_per_byte()` is an exact number, not an estimate (see
      `docs/architecture.md`'s bpelm section for how this was caught and fixed).
- [x] **Exploratory 25000-step run, `qcute/bytelm`** at the *original* K=8-
      equivalent bandwidth (`mtp_heads=8`, before the bandwidth target was
      recalibrated — see below): val_bpb bottomed at **~2.55** around step
      1500, then rose non-monotonically past the unigram range (~4.5–5.0) to
      **6.43** by step 25000, while train bpb collapsed to ~0.2–0.3 — classic
      overfitting on a tiny dataset, made *visible* specifically because LR
      was held constant rather than decayed (`docs/scaffolding_playbook.md`
      §7) — also the textbook setup for epoch-wise double descent (Nakkiran
      et al. 2019); no second descent showed up within the 25000-step budget.
      Self-speculative decoding (MTP heads as draft) on the final checkpoint:
      7.08x speedup over plain AR, avg accept length 1.78/8.
- [x] **Bandwidth target recalibrated to 4 bytes/timestep** for the tiny-
      corpus regime (`bytelm`'s `xs` preset: `mtp_heads` 8→4; `qcutelm`'s `K`
      8→4) after the BPE baseline empirically showed only ~3–4 bytes/token
      is achievable on a 450KB corpus before larger vocabs start memorizing
      phrases instead of generalizing — targeting 8 was a bandwidth mismatch
      specific to this data scale, not the doc's original target being wrong
      (see `docs/scaffolding_playbook.md` §8d). `sd`/`md` keep 8, matching
      the doc's full-corpus-scale default.
- [x] **Matched-bandwidth comparison, all three baselines, 2000 steps each**,
      same tiny 500KB subset (superseded — see the enwik8_1M.gz rerun
      below), same warmup+constant LR schedule:

  | baseline | run | best val_bpb | at step | params |
  |---|---|---|---|---|
  | `bytelm` (mtp_heads=4) | `bytelm_xs_mtp4` | **2.52** | 1300 | 3.4M |
  | `bpelm` (vocab=8192) | `bpelm_8192` | **2.35** | 300 | 5.25M |
  | `qcutelm` (BSQ, K=4) | not yet re-run at K=4 (below is the older K=8 run) | 4.97 | 4750 | 3.9M |

  Plots: `logs/bytelm_xs_mtp4/bpb.png`,
  `logs/bpelm_8192/bpb.png` (`scripts/plot_run.py`). Both bytelm
  and bpelm show the same shape — a clean bottom, then overfitting as train
  bpb collapses toward zero — but bpelm gets there **much faster** (bottoms
  at step 300 vs. bytelm's 1300, and train bpb reaches ~0.02 by step 2000
  vs. bytelm's ~1.3), consistent with bpelm's much bigger effective
  vocabulary (8192 tokens vs. 256 bytes) memorizing a small corpus quicker.

- [x] **Rerun on the standardized enwik8_1M.gz corpus** (`--config
      configs/bytelm_xs_mtp4.py` / `configs/bpelm_8192.py`, both 2000
      steps, both checkpoints saved to `checkpoints/{bytelm_xs_mtp4,
      bpelm_8192}/{best,last}.pt`):

  | baseline | best val_bpb | at step | end-of-run state |
  |---|---|---|---|
  | `bytelm_xs_mtp4` | **2.4872** | 2000 | clean — train_bpb≈1.9, no overfitting red flags, `best.pt`≈`last.pt` |
  | `bpelm_8192` | **2.3679** | earlier (before drift) | **overfit by the end** — train_bpb collapsed to ~0.34-0.54 while val_bpb was actively *climbing* (3.13→3.24→3.35 over the final 3 evals), diverging not just plateauing |

  Same qualitative pattern as the 500KB-subset run (bpelm overfits faster/
  harder than bytelm), confirmed again on the full corpus. **Caveat for any
  comparison against these numbers**: bpelm's `best.pt` (val_bpb 2.3679) is
  the only trustworthy checkpoint — `last.pt` reflects a badly overfit model
  and its train_bpb is not a meaningful generalization signal at all.
  bytelm's full trajectory (train and val) is usable as-is.
  **bpelm currently has the best number of the three** at this scale, but
  take that with real caution: it's the highest-capacity model (5.25M vs.
  ~3.5-3.9M) and its overfitting is also the fastest, so this isn't yet a
  fully controlled comparison (param count and overfitting rate both differ,
  not just bandwidth mechanism).
- [ ] Go/no-go (baselines reproduce literature numbers within ±5%) — not
      applicable; no published byte/BPE-on-enwik8-tiny-subset number to
      compare against. The real Phase 0 go/no-go (full enwik8, `sd`/`md`
      presets, mtp_heads=8) hasn't been attempted — everything above is on
      the 500KB tiny subset for fast local iteration.

## Phase 1 — standalone autoencoder

Superseded by an end-to-end approach (see Phase 2 below) before this phase's
go/no-go was cleared. Old streaming-causal-encoder implementation archived
at `archive/tokenizer_phase1_standalone_autoencoder.py` (never trained to
convergence either — smoke-tested only).

## Phase 2 — LM in latent space

- [x] Encoder + FSQ/BSQ bottleneck + latent LM + decoder, trained jointly,
      with a generation loop, interface Option A (`qcute/qcutelm.py`).
      Simplified vs. the doc: non-streaming chunk-local MLP encoder/decoder
      instead of the recommended causal-SSM designs — see
      [architecture.md](architecture.md) for the full list of simplifications.
- [x] **Training runs, `qcute/qcutelm` (BSQ)**, same tiny 500KB subset, same
      LR schedule as bytelm, at the *original* K=8: 5000 steps
      (`configs/qcutelm_bsq_tiny.py`) gave best val_bpb **4.97** (step 4750,
      still gently improving at the end); extended to 25000 steps (matching
      bytelm's original budget) showed the same overfitting shape as bytelm
      — smooth decrease to ~5.0 by step ~3500–5000, then a non-monotonic
      rise to ~6.5 by step ~14500 — just delayed, roughly consistent with
      training **~10x faster per step** than bytelm's `xs` (0.044s/step vs.
      0.47s/step; chunk-local MLP encoder/decoder is much cheaper than a
      full byte-level causal transformer pass) needing more steps for
      comparable *wall-clock* exposure.
- [x] **Re-run at recalibrated K=4** (see Phase 0's bandwidth-recalibration
      entry), loosely-coupled architecture (old, still the FSQ default):
      2000 steps hadn't converged (val_bpb still falling at the end);
      extended run, stopped at step ~6300: best val_bpb **5.54** (step
      5400), val_bpb still ~5.56 at the stopping point — clearly still
      slower to converge in step-count than bytelm/bpelm at this scale
      (5400 vs. 1300/300), consistent with the earlier K=8 finding.
      `logs/qcutelm_bsq_k4/bpb.png`.
- [x] **Architecture change: BSQ's default training path is now tightly
      coupled** — `encoder → z_t → LM → raw latent → bsq_quantize →
      decoder → bytes_{t+1}`, graded against the true next bytes, instead
      of the old loosely-coupled "decoder reconstructs the encoder's own
      code" (still FSQ's behavior, and available for BSQ too via
      `cfg.aux_recon`, on by default, `--disable_aux_recon` to turn off).
      See `docs/architecture.md`'s qcutelm section for the full design and
      why (LM's input stays a continuous linear projection, not a discrete
      embedding table — the implicit BSQ codebook, `2^18` at the default
      `dq=18`, would need a ~67M-param table; a factorized/PQ-style
      embedding was considered and deliberately deferred).
  - **Result** (`aux_recon=False`, K=4, 10000 steps): plateaus at val_bpb
    **~8.0–8.5** after a brief dip to ~7.1 near step 400 — substantially
    worse than the loosely-coupled architecture's 5.54. Notably, train and
    val bpb track *closely together* the whole run (no growing gap) — this
    is **not overfitting**, unlike every other run so far; it looks stuck
    in a poor local optimum instead. `logs/qcutelm_bsq_k4/bpb.png`.
  - **`--lfq` added**: regress BSQ's quantizer to plain LFQ (Yu et al.
    2023) — skip the L2-normalize-onto-hypersphere step, sign the raw
    projection directly (hypercube corners `{-1,+1}^dq`, unscaled) instead
    of BSQ's hypersphere corners (`||z_hat||=1`, scaled `1/sqrt(dq)`). Sign
    bits (targets) are identical either way, only `z_hat`'s geometry
    changes; wired through `bsq_quantize`/`Config.lfq`/CLI.
  - **Result, `--lfq --aux_recon` (default on), K=4, 10000 steps**
    (`qcutelm_bsq_k4_lfq_aux`): best val_bpb **6.01** at step 6000 — a big
    recovery from the no-aux BSQ plateau (~8.0–8.5), though still short of
    the loosely-coupled architecture's 5.54. `latent_acc` (LM predicting
    the right sign bits) saturates near 99.5–100% almost immediately —
    the LM side is essentially solved. `recon_acc`/`aux_recon_acc` (decoder
    reconstructing bytes from `z_pred`/`z_hat`) stay stuck around 5–11%
    the whole run — **the decoder is now the bottleneck**, not the LM,
    and notably `aux_recon_acc` (decoding the encoder's *own*, presumably
    "easier," code) is consistently a bit *lower* than `recon_acc` (decoding
    the LM's predicted code) — plausibly because one shared decoder has to
    serve two different input code distributions (`z_hat` from the encoder,
    `z_pred` from the LM) at once. `logs/qcutelm_bsq_k4_lfq_aux/bpb.png`.
  - **Confound resolved: `qcutelm_bsq_k4_aux`** (plain BSQ, no `--lfq`,
    `aux_recon=True`, K=4, 10000 steps) — the LFQ swap and the aux_recon
    swap turn out to pull in *opposite* directions, not the same one.
    Over the run, `val_aux_recon_acc` climbs steadily 13%→70% (the decoder
    gets *much* better at decoding the encoder's true code — far better
    than LFQ+aux's 5%) while `val_latent_acc` steadily *falls* 89%→~59%
    (the LM gets *worse* at predicting the next code, unlike LFQ+aux's
    ~99.7%). Net effect: `val_bpb` gets *worse* over training (7.29 at
    step 200 → ~8.4–8.6 by step 7000+, i.e. the checkpointer's "best" is a
    misleading early accident, not a converged optimum — same trap as the
    no-aux run's step-400 "best"). `logs/qcutelm_bsq_k4_aux/bpb.png`.
  - So: **BSQ's hypersphere-normalized codes are more decodable but harder
    for the LM to predict; LFQ's raw hypercube-corner codes are trivial for
    the LM but much less decodable** — a real geometry tradeoff, not one
    knob dominating the other. Neither beats the loosely-coupled
    architecture's 5.54 yet.
  - **Encoder reverted to plain MLP**: a causal-TCN encoder (dilated conv +
    early-byte-weighted pooling) was tried in between these runs, then
    reverted — it was solving a problem in the wrong place. The doc's own
    design has the decoder be **non-causal one-shot** because `z_t`
    (LM-produced) already carries cross-chunk context (handover doc, "decoder
    stays non-causal one-shot... doesn't need access to prior bytes"); within
    a fully-observed K-byte chunk there's no correctness reason for the
    encoder to hide future bytes from earlier positions either, and the
    causal structure actively fought the early-byte-weighted pooling it was
    paired with (see git history for `CausalConvBlock`/`enc_decay` — not
    kept). Causality belongs to the LM (across chunks), not the chunk-local
    encoder/decoder.
  - **`--pretrain_ae` added**: optional encoder+decoder-only warm-start
    phase before joint training (own AdamW/LR, LM untouched), training
    plain reconstruction CE until val recon_acc clears `--pretrain_target_acc`
    (default 95%) or `--pretrain_steps` (default 3000) is hit. Saves an
    encoder/decoder-only checkpoint (`checkpoints/<run_name>/pretrain_ae.pt`);
    `--init_encoder_decoder_from <path>` warm-starts a *different* run's
    joint training from a previously-saved one. Motivation: give the decoder
    a head start on the true-code mapping before the LM (and its noisy
    `z_pred`) enters the picture, since the decoder has been the bottleneck
    in every tightly-coupled variant so far.
  - **Result, `qcutelm_bsq_k4_pretrain_ae`** (plain BSQ, K=4, 3000-step
    pretrain cap, then 10000-step joint training): pretraining plateaued at
    **70.1% recon_acc**, hitting the 3000-step cap without reaching the 95%
    target. Joint training then converged to essentially the *same* place
    as the non-pretrained `qcutelm_bsq_k4_aux` run — best val_bpb **8.32**
    at step 7600, `latent_acc` ~60%, `aux_recon_acc` ~74–80% — no better
    than training from scratch. **The head start didn't survive joint
    training**: with all params unfrozen during the joint phase, the LM's
    `pred_loss` reshapes `z_hat` again and pulls the encoder/decoder away
    from the pretrained solution rather than preserving it. Pretraining
    encoder/decoder alone doesn't fix the underlying LM-predictability vs.
    decoder-quality tension — that tension is reintroduced the moment joint
    training resumes. `logs/qcutelm_bsq_k4_pretrain_ae/bpb.png`. A frozen or
    slow-unfreeze warm-start (encoder/decoder LR held near zero for the
    first N joint steps) might behave differently — not yet tried.
  - **Diagnosis (`scripts/diagnose_qcutelm.py`, new)**: per-position (byte
    0..K-1) recon_acc breakdown, and per-loss-term gradient norm
    (`||grad||` over all params, one batch, each of `rec_loss`/`pred_loss`/
    `aux_rec_loss` backpropped separately) for any BSQ checkpoint.
    - **Per-position**: no meaningful byte-position degeneracy found in
      either `qcutelm_bsq_k4_lfq_aux` or `qcutelm_bsq_k4_aux` — all 4
      positions land within a few points of each other. The earlier
      "weight early bytes more" hypothesis (before it was reverted) wasn't
      addressing a real problem.
    - **Gradient norms — the actual smoking gun**: at `qcutelm_bsq_k4_lfq_aux`
      step 6000, `aux_rec_loss`'s gradient norm is **26.2M**, vs. `rec_loss`'s
      428K (~60x smaller) and `pred_loss`'s 4.2K (~6000x smaller). At
      `qcutelm_bsq_k4_aux` (plain BSQ) step 10000, all three are small and
      balanced (0.45 / 0.21 / 0.20) — no dominance at all. The difference is
      **code magnitude**: LFQ's `z_hat` is raw `±1` per dim (`||z_hat|| =
      sqrt(dq) ≈ 4.24` at dq=18), while BSQ's is L2-normalized then scaled
      by `1/sqrt(dq)` (`||z_hat|| = 1`, always). That ~18x larger input
      compounds through the decoder's linear layer into far larger
      pre-softmax logits and gradients. Since training does one combined
      `loss.backward()` + global grad-clip (all three terms summed before
      clipping to norm 1.0), LFQ's `aux_rec_loss` gradient direction
      dominates essentially every update — plausibly explaining why LFQ's
      `aux_recon_acc` (5%) ends up *worse* than plain BSQ's (~70%) despite
      nominally getting the dominant gradient share: a huge, poorly-scaled
      gradient direction isn't the same as a *useful* one. This also ties
      together LFQ's other effect (near-trivial LM prediction, 99.7%
      `latent_acc`) — the same large, easily-separable `±1` targets that
      make BCE easy for the LM are what blow up the decoder's gradient
      scale. **Candidate fix, not yet tried**: explicit loss-term weighting
      (downweight `aux_rec_loss`/`rec_loss` relative to `pred_loss`), or a
      fixed (non-adaptive) `1/sqrt(dq)` rescale on LFQ's `z_hat` to tame its
      magnitude without reintroducing BSQ's per-example L2-normalization.
  - **Qualitative generation comparison (`scripts/qualitative_compare.py`,
    new)**: same 6 raw-byte prompts (3 train-region, 3 val-region) fed to
    `bytelm_xs_mtp4`, `bpelm_8192`, and
    `qcutelm_bsq_k4_lfq_aux`, 64-byte continuations, ground truth from the
    real corpus alongside each. **Result: `qcutelm` outputs the exact same
    degenerate `' o a o a o a...'` string for all 6 prompts**, completely
    ignoring prompt content — not "worse English," literal generation
    collapse into a 2-byte cycle. `bytelm`/`bpelm` both produce recognizable
    (if imperfect/nonsensical) wiki-markup English — real words, correct
    `[[...]]` link syntax — never degenerate. This is a much starker gap
    than the bpb numbers alone suggest (qcutelm's val_bpb is ~2.5x
    bytelm/bpelm's, which reads as "worse," not "broken"). Root cause is
    almost certainly the same decoder bottleneck already diagnosed above:
    BSQ's `generate()` is fully deterministic (`bsq_quantize` always takes
    the hard sign, no temperature/sampling unlike FSQ's softmax path), so
    a decoder/LM pair this inaccurate settles into a fixed 2-state limit
    cycle regardless of the conditioning latent. Output saved to
    `logs/qualitative_compare/compare_*.{txt,json}`.
  - **All K=4 BSQ variants side by side** (best-val_bpb's step; `recon_acc`
    is the decoder's exact-match byte accuracy at that step, `z_pred`-based
    except where noted):

    | run | architecture | best val_bpb | at step | recon_acc | latent_acc |
    |---|---|---|---|---|---|
    | `qcutelm_bsq_k4` | loosely-coupled (historical) | 5.54 | 5400 | 72.76% | 60.66% |
    | `qcutelm_bsq_k4` | tightly-coupled, no aux | 7.12* | 400 | 13.48% | 90.34% |
    | `qcutelm_bsq_k4_lfq_aux` | tightly-coupled, LFQ + aux | 6.01 | 6000 | 9.44% (aux: 5.12%) | 99.73% |
    | `qcutelm_bsq_k4_lfq_aux_uw` | tightly-coupled, LFQ + aux + uncertainty weighting | **5.39** | 8600 | 12.21% (aux: 9.69%) | 99.78% |
    | `qcutelm_bsq_k4_aux` | tightly-coupled, BSQ + aux | 7.29* | 200 | 13.36% (aux: 13.31%) | 89.37% |
    | `qcutelm_bsq_k4_pretrain_ae` | tightly-coupled, BSQ + aux + AE pretrain | 8.32* | 7600 | 25.78% (aux: 74.37%) | 60.48% |
    | `qcutelm_bsq_k4_dq14_aux` | tightly-coupled, BSQ + aux, dq=14 (was 18) | 6.74* | 200 | 21.01% (aux: 64%) | 59.47% |

    \*the no-aux and BSQ+aux rows' logged "best" checkpoints are early
    accidents (steps 400/200) before their real dynamics kick in — not
    representative of converged quality, unlike the other two rows; BSQ+aux
    actually gets *worse* in val_bpb as training proceeds (see above) even
    as `aux_recon_acc` climbs to ~70% by step 10000. Note the inverse
    relationship between `latent_acc` and decoder quality across rows: the
    loosely-coupled run has the worst LM (60.66%) but a strong decoder
    (72.76%); LFQ+aux has a near-solved LM (99.73%) but the worst decoder
    (9.44%); BSQ+aux sits in between and gets *worse* over time — no
    variant has both a good LM and a good decoder simultaneously yet.
  - **`dq=14` (down from 18, chosen as `2^14=16384`, just above BPE's 8192
    vocab)**, plain BSQ + aux: same misleading-early-accident "best" (6.74
    at step 200), but a real converged-quality improvement over dq=18 —
    val_bpb settles around **~7.8** by step 9800-10000, vs. dq=18's ~8.4–8.6.
    Mostly a mechanical effect, though, not evidence of better training
    dynamics: `bpb_lm_only = pred_loss * dq / (K*ln2)` scales linearly with
    `dq`, so shrinking `dq` directly shrinks that term regardless of whether
    anything actually got easier to learn — and indeed `latent_acc` (~59%)
    and `aux_recon_acc` (~64%, if anything *lower* than dq=18's ~70%) look
    about the same as dq=18. `logs/qcutelm_bsq_k4_dq14_aux/bpb.png`.
  - **Uncertainty weighting added** (`--uncertainty_weighting`, BSQ only):
    Kendall & Gal (2018)-style learned homoscedastic weighting on
    `rec_loss`/`aux_rec_loss` — one trainable `log_var` scalar per loss,
    each reweighted as `exp(-log_var)*loss + log_var` instead of a fixed
    1x coefficient; `pred_loss` (BCE, different loss family/scale) stays
    unweighted. Lives in the **training loop** (`main()`), not on
    `QCuteLM` — it's a loss-combination choice, not model architecture;
    `model.forward()` still returns the plain unweighted sum, and the
    actual backward loss is recomputed from the raw per-term losses in
    `metrics` only when the flag is on. Directly motivated by the
    diagnosed gradient-scale imbalance above.
  - **Result, `qcutelm_bsq_k4_lfq_aux_uw`** (LFQ + aux + uncertainty
    weighting, K=4, 10000 steps): best val_bpb **5.39** at step 8600 — beats
    every other qcutelm result so far, *including* the loosely-coupled
    architecture's 5.54. **Caveat, important**: `log_var_rec`/`log_var_aux`
    barely moved from their 0.0 init (ended at 0.015 both), meaning the
    learned weight `exp(-log_var)` stayed at ≈0.985 — essentially
    unchanged from the unweighted 1x baseline. So this result is *not*
    good evidence the uncertainty-weighting mechanism is what helped —
    training has no fixed random seed (`batch_iter` uses unseeded
    `torch.randint`), so this may simply be run-to-run variance landing in
    a better basin. **Needs a rerun without `--uncertainty_weighting`
    (or with seeding controlled) to actually isolate the effect** before
    treating 5.39 as attributable to the technique. `logs/qcutelm_bsq_k4_lfq_aux_uw/bpb.png`.
  - **Variance check, `qcutelm_lfq_r2`**: same config as `qcutelm_bsq_k4_lfq_aux`
    (LFQ+aux, no uncertainty weighting), rerun independently (unseeded) —
    best val_bpb **6.18**, close to the original **6.01**. Two independent
    non-uncertainty-weighted runs landing near each other (6.01, 6.18)
    while the uncertainty-weighted run got notably lower (5.39) reads as a
    real gap, not just noise — though since `log_var` barely moved in that
    run, whether the *adaptive-reweighting mechanism itself* is what
    caused it (vs. e.g. the extra parameters/gradient-graph shape changing
    the optimization trajectory some other way) is still unconfirmed.
    `logs/qcutelm_lfq_r2/bpb.png`.
  - **Entropy regularization added** (`bsq_entropy_reg`, `--entropy_reg_weight`):
    the LFQ/BSQ-paper entropy term (Yu et al. 2023 §3.2) — minimize
    per-example bit entropy, maximize batch-averaged bit-usage entropy —
    identified as a real, literature-documented gap in this project's BSQ
    implementation (neither paper's published recipe works without it; see
    the literature-contrast discussion this session). Computed on the LM's
    raw predicted latent `v_pred` (not the encoder's), since that's where
    the qualitative-generation collapse actually lives. Lives in the
    training loop like uncertainty weighting, not the model; when both
    flags are on, `entropy_reg` gets its own learned `log_var` too.
  - **Result, `qcutelm_lfq_aux_uw_ereg`** (LFQ + aux + uncertainty
    weighting + entropy_reg, K=4, 10000 steps, pre-MaskGIT decoder): worse
    than uncertainty weighting alone — best val_bpb 7.35 (misleading early
    accident, same pattern as other runs), converging to **~8.08** by step
    10000, notably worse than uw-alone's 5.39. `latent_acc` even dips to
    88–91% at times late in training (worse than uw-alone's near-100% —
    the LM got *less* stable, not more). **Likely mechanism**:
    `bsq_entropy_reg` has no knowledge of the true target bits — it only
    rewards confident-per-example + diverse-batch-average predictions,
    which can be satisfied in ways uncorrelated with (or actively opposed
    to) `pred_loss`'s job of matching the correct sign. Combining an
    unconstrained, correctness-blind regularizer with uncertainty
    weighting (which adaptively chases raw gradient *scale*, not
    *usefulness*) let entropy_reg's large raw magnitude (its natural range
    is `±dq·ln2 ≈ ±12.5` at dq=18, vs. CE losses in the 3-6 range) pull
    training somewhere entropy-favorable but correctness-*un*favorable.
    `logs/qcutelm_lfq_aux_uw_ereg/bpb.png`. **Takeaway**: entropy
    regularization is a real, doc/literature-motivated idea, but this
    naive wiring (unweighted-by-default, or freely uncertainty-weighted)
    isn't the right way to combine it with the other fixes — a small
    fixed coefficient (not uncertainty-weighted) or a warmup schedule is
    likely needed; not yet tried.
  - **MaskGIT-style decoder implemented** (handover §1.4b): `ChunkDecoder`
    rewritten from "predict all K bytes independently from `z` alone" (an
    implicit conditional-independence assumption across the K positions)
    to seeing a partially-masked byte chunk (`byte_emb(x_masked) + pos_emb
    + z_proj(z)` → MLP → per-position logits over real bytes), trained
    with a cosine mask-rate schedule (`maskgit_mask`, Chang et al. 2022)
    and decoded at inference with `--maskgit_T` (default 4) confidence-
    based refinement steps (`maskgit_decode`) — directly targets the
    decoder-capacity bottleneck implicated across most of this section
    (independent-position decoding, unable to model intra-chunk byte
    correlations). Touches every decoder call site (`forward()` FSQ/BSQ
    paths, `generate()`, `score_continuation_bpb()`,
    `pretrain_autoencoder()`, `eval_ae_recon_acc()`,
    `scripts/diagnose_qcutelm.py`) — old checkpoints are architecturally
    incompatible (decoder's parameter shapes changed) and can't be loaded
    anymore. Note the resulting `bpb_rec`/`bpb_total` is now a heuristic
    proxy (masked CE over a random per-batch subset of positions), not a
    tight ELBO — the doc's own caveat for MaskGIT (§1.4b) vs. its
    recommended time-free-masked-diffusion variant (not implemented).
    First training attempt (`qcutelm_maskgit_lfq_aux_uw_ereg`, stacking
    MaskGIT + uncertainty weighting + entropy_reg all at once) diverged
    outright — `latent_acc` collapsed to ~43-44%, val_bpb blew up to ~22.6
    by step 4600, same entropy_reg-driven instability as the
    non-MaskGIT `qcutelm_lfq_aux_uw_ereg` run above, compounding with the
    new decoder. Killed early (logs kept as
    `logs/qcutelm_maskgit_lfq_aux_uw_ereg_diverged/`) — stacking three
    unvalidated changes into one run was the mistake, the same lesson as
    the LFQ-vs-aux_recon confound earlier in this doc. **Rerun with just
    MaskGIT decoder alone** (LFQ+aux, no uncertainty weighting, no
    entropy_reg — `qcutelm_maskgit_lfq_aux`): stable this time (no
    divergence), best val_bpb **5.91** at step 9800, `latent_acc` ~97%.
    `recon_acc` (13.3%) is meaningfully better than the pre-MaskGIT LFQ+aux
    baseline's ~9–11%, confirming the decoder-side fix does help on its
    own — but bpb only nudged down slightly (6.01→5.91), still short of
    the loosely-coupled baseline's 5.54 and the uncertainty-weighted
    (pre-MaskGIT) run's 5.39. `logs/qcutelm_maskgit_lfq_aux/bpb.png`.
  - **A second, bigger decoder bug found and fixed: zero cross-position
    mixing.** The MaskGIT decoder above still only modestly improved
    `recon_acc` (13% vs. 9%) — investigating why turned up that
    `ChunkDecoder.forward()` applied a plain per-position
    `Linear`/`GELU`/`Linear` MLP to `[N, K, d_byte]`. Since `nn.Linear`
    broadcasts over all leading dimensions, this was **literally zero**
    mixing between the K positions — masked positions could never actually
    see unmasked ones, despite the docstring's claim otherwise and despite
    MaskGIT's entire premise depending on exactly that. Fixed with
    `FullSelfAttention` (real non-causal self-attention over the K
    positions) or `FullConvMixer` (single non-causal conv,
    `kernel_size=2K-1`, symmetric `padding=K-1` — verified via gradient-
    flow check that every output position now sees all K inputs; an
    earlier `kernel_size=K` version was also tried and found wrong, same
    class of bug, smaller scale — see architecture.md). Also added
    `--disable_pred_loss` (drop the BCE code-supervision term entirely,
    letting `rec_loss`'s STE gradient alone shape the LM), `--lm_factorized_input`
    (`FactorizedCodeEmbedding`, a PQ-style compositional embedding — see
    below), and matched `qcutelm.py`'s three `AdamW` instances to
    `bytelm.py`/`bpelm.py`'s tuned settings (`betas=(0.9,0.95),
    weight_decay=0.1`, previously plain PyTorch defaults).
  - **Result, `qcutelm_joint_bsq_noaux_fixedmix`** (plain BSQ, no aux, no
    uncertainty weighting/entropy_reg — the exact config of the original
    `qcutelm_bsq_k4` no-aux baseline, just with the fixed
    decoder): stabilizes at `recon_acc` **~24-26%** (a real plateau from
    step ~5000 on, not still climbing) and val_bpb **~6.9-7.0** — vs. the
    pre-fix baseline's permanent ceiling of 13.48% `recon_acc` and ~8.0-8.5
    bpb that never budged. Roughly **2x** the reconstruction accuracy and
    a real, stable (not accidental-early-dip) improvement — strong direct
    evidence the mixing bug was a major cause of the decoder-bottleneck
    finding that recurred across most of this section, though 24-26% is
    still far from loosely-coupled's historical 72.76%, so it wasn't the
    *whole* story either. `logs/qcutelm_joint_bsq_noaux_fixedmix/bpb.png`.
  - **Result, `qcutelm_joint_bsq_noaux_nopred_30m`** (same config plus
    `--disable_pred_loss`, pure `rec_loss`-only training, extended to a
    30-minute/~46000-step budget to actually watch for an overfitting
    onset instead of assuming the fixed 10000-step convention was long
    enough): `recon_acc` climbed faster early than with `pred_loss` on,
    but **plateaus around val 32-35% from step ~25000 onward** — a real
    plateau, not still-improving. `latent_acc` stayed ~50-51% throughout
    (vs. ~80% with `pred_loss` on) and `bpb_lm_only` ~10 (vs. ~2.5-2.8) —
    not comparable to the `pred_loss`-on run's bpb, since `bpb_pred` is
    derived from the now-uncalibrated raw BCE value, not decode quality
    (see architecture.md). A **mild, late-onset train/val gap** appears
    only near the very end (train `recon_acc` reaching 37-41% vs. val's
    32-35% by step ~44000-46000) — the beginning of overfitting, not
    severe yet. `logs/qcutelm_joint_bsq_noaux_nopred_30m/bpb.png`.
  - **Result, `qcutelm_joint_bsq_noaux_nopred_dq14_30m`**: same ablation
    at `dq=14` instead of 18 — plateaus at `recon_acc` ~31-33% (max 33.6%
    at step 45000), **essentially the same plateau as dq=18** (32-35%),
    not a meaningful improvement despite converging a bit faster early on.
    `bpb_total` looks much better (7.2 vs. 13.9) but that's mechanical —
    `bpb_pred` shrinks with `dq` regardless of quality, and it's already
    uncalibrated garbage under `--disable_pred_loss` (see the earlier
    caveat) — not a real signal. Took ~2.5 hours wall-clock instead of the
    intended ~30 min (throughput dropped partway through, cause not yet
    investigated — worth checking system load next time before assuming a
    long run will finish on schedule). `logs/qcutelm_joint_bsq_noaux_nopred_dq14_30m/bpb.png`.
    Since `dq=14` didn't clearly help, trying `dq=13` (`2^13=8192`, exact
    match to bpelm's vocab size) next per the fallback plan — though given
    `dq` clearly isn't the dominant lever here (18→14 changed nothing
    meaningful), `dq=13` is unlikely to differ much either; if it doesn't,
    `dq` should be deprioritized as a variable and effort redirected to
    the aux_recon/PQ-embedding/mixer branches of the grid instead.
  - **Result, `qcutelm_joint_bsq_noaux_nopred_dq13_30m`**: confirmed —
    plateaus at `recon_acc` ~30-32% (max 32.5% at step 46000), essentially
    identical to both dq=14 (~31-33%) and dq=18 (~32-35%). **`dq` (across
    13/14/18) is not the dominant lever for this bottleneck** —
    deprioritized as a variable going forward; the plateau is coming from
    somewhere else (likely `aux_recon`/`pred_loss` presence, or the mixer
    itself, or the corpus-size ceiling). Also took even longer this time
    (~3.5 hours wall-clock for the same ~46000-step budget) — the
    throughput slowdown is a **recurring pattern across all three of
    these long runs, not a one-off fluke**; worth investigating
    separately (thermal throttling, memory growth over very long MPS
    runs, or background system load are the likely candidates) before
    launching further multi-hour runs. `logs/qcutelm_joint_bsq_noaux_nopred_dq13_30m/bpb.png`.
  - **Result, `qcutelm_joint_bsq_aux_pq_nopred`** (PQ/factorized LM input,
    aux_recon **on**, `pred_loss` still off, dq=18, 15000 steps): max
    `val_recon_acc` **28.8%** at step 14400 — still climbing at the end,
    not plateaued, unlike every no-aux dq variant above (which fully
    plateaued by a similar or larger step count). `val_aux_recon_acc`
    (decoding the encoder's *true* code) climbed to **~70-74%** then
    plateaued around step ~5000 — a real, stable ceiling on that easier
    sub-task, well above any no-aux run's `recon_acc`. `logs/qcutelm_joint_bsq_aux_pq_nopred/bpb.png`.
    Next: rerun with `pred_loss` enabled to isolate that variable, and if
    nothing in this grid gets even *train*-set `recon_acc`/`aux_recon_acc`
    close to 100%, run a fast tiny-subset `--pretrain_ae`-only sanity
    check (no LM at all) — if the tokenizer alone can't overfit a tiny
    subset to ~100%, that's a real architecture/capacity bug, not an
    optimization or corpus-scale issue, and points straight at trying a
    bigger encoder/decoder next.
  - **Sanity check run, root cause found: tokenizer depth, not a bug.**
    `ChunkEncoder`/`ChunkDecoder` had exactly **one** `MixerBlock` hardcoded
    — never stacked, regardless of width. On a tiny 20000-byte subset,
    `--pretrain_ae`-only (no LM) capped at train `recon_acc` ~88-97%
    depending on width/LR/weight-decay tuning, **never reaching 100%** no
    matter how those were tuned (tried: 4x steps, half LR, wider dims
    `d_byte=128/d_enc=d_dec=512`, `weight_decay=0`, warmup→constant→cosine
    LR decay — none of it closed the gap alone). Added `cfg.tokenizer_layers`
    (`--tokenizer_layers`, both `ChunkEncoder`/`ChunkDecoder` now stack N
    `MixerBlock`s) — with **2 layers**, on a smaller 2000-byte subset, train
    `recon_acc` reaches **exactly 100.00%** (loss=0.0000) for both mixers:
    conv at step 3594, attention at step 3398 — close, not dramatically
    different. Confirmed stable, not a fluke: attention held exactly
    100.00% (loss=0.0000) from ~step 17000 through the full 20000-step
    run, not a transient spike. `val_recon_acc` plateaus ~70-78% either
    way (expected — a genuine train/val gap on such a tiny dataset, not a
    concern for this specific test). **Conclusion: the plateau across this
    whole grid was
    partly an undercapacity tokenizer (1 layer, insufficient depth) — every
    joint-training experiment above used this same 1-layer tokenizer.**
    Also added in the process: `--d_byte`/`--d_enc`/`--d_dec` (previously
    hardcoded, no CLI override existed at all), `--pretrain_weight_decay`,
    `--pretrain_cosine_decay` + `--pretrain_constant_steps`
    (warmup→constant→cosine schedule, `lr_at_warmup_constant_cosine`), and
    an encoder/decoder/LM param-count breakdown in the startup log line
    (previously only the total was logged, hiding that the LM — not the
    tokenizer — is what actually dominates the ~3.4M-param total; the
    tokenizer alone is only ~0.15-0.25M params).
  - **Open question, not yet answered**: does `tokenizer_layers=2` stay
    sufficient at the *actual* corpus scale (450K train bytes), or does the
    capacity requirement scale up with dataset size the way it did going
    from 20000→2000 bytes here? The joint-training grid below needs
    rerunning with `tokenizer_layers=2` before its earlier (1-layer)
    results can be trusted as representative of the architecture's real
    ceiling.
  - **Full experiment grid now in flight** (fixed-decoder retest era):
    `{dq: 18, 14, 13} × {mixer: conv, attention} × {pred_loss: on, off} ×
    {aux_recon: on, off} × {lm input: continuous, factorized/PQ}`. Not
    exhaustively swept — worked through in priority order, one variable
    isolated at a time (established practice this session), pruning
    branches the results so far suggest don't matter. Planned order: (1)
    finish the `dq` sweep on the no-aux/no-pred-loss/conv baseline, (2)
    PQ-embedding + aux_recon, with and without `pred_loss`, (3) LFQ+aux,
    uncertainty weighting, entropy regularization retests, all with the
    fixed decoder, (4) repeat the highest-value combinations from (1)-(3)
    with `--mixer attention` instead of `conv` for a full comparison, (5)
    fallback: if nothing above clears a real bar, retry the frozen-
    tokenizer pipeline requiring train/val `recon_acc` >=90% match, or —
    if even that fails — warm-start joint training from whatever weak
    tokenizer pretraining does reach (`--init_encoder_decoder_from`)
    rather than random init.
  - **Rule added to CLAUDE.md**: only ever run one training job at a time
    — running two concurrently on MPS was observed directly to stall one
    of them at zero throughput (contention), not just slow both down
    proportionally.
- [ ] Go/no-go (BPB matches or beats BPE+MTP baseline at matched compute) —
      at the now-comparable 4-bytes/timestep numbers: bpelm (2.35) < bytelm
      (2.52) < qcutelm tightly-coupled lfq+aux+uncertainty-weighting (5.39,
      caveated — see above) < qcutelm loosely-coupled (5.54) < qcutelm
      tightly-coupled lfq+aux (6.01) < qcutelm tightly-coupled BSQ+aux (7.29,
      misleading early-accident "best"; converges to ~8.4–8.6) ≈
      BSQ+aux+AE-pretrain (8.32, no better than without pretraining) ≈
      BSQ+aux+dq14 (~7.8, mostly mechanical) < qcutelm tightly-coupled BSQ,
      no-aux (~8.0–8.5 plateau). qcutelm is currently the weakest of the
      three at this scale by every architecture variant tried so far,
      though the LFQ+aux+uncertainty-weighting result is (unverified
      caveat aside) the closest it's gotten. Qualitative generation
      (`scripts/qualitative_compare.py`) is arguably the more damning
      signal right now: qcutelm's best checkpoint (lfq_aux, pre-
      uncertainty-weighting) collapses into a fixed repeating byte pattern
      regardless of prompt, while bytelm/bpelm both produce recognizable
      wiki-markup English — bpb numbers alone understate how far apart
      generation quality actually is.
- [x] **Alternative "dumber, simpler" pipeline added**: pretrain encoder+decoder
      (`--pretrain_ae`) until good, **freeze** them (`--freeze_after_pretrain`,
      new — the missing piece vs. plain `--pretrain_ae`, which let joint
      training erase the pretrained solution), then build a **discrete
      vocabulary from the frozen tokenizer's actual codes** (`build_code_vocab`
      — dedups on `targets`, like BPE builds a vocab from merges, not the
      full combinatorial codebook) and train a **separate, plain categorical
      LM** (`train_vocab_lm`) — real `nn.Embedding` + weight-tied softmax
      head over that vocab, exactly like `qcute.bpelm`'s architecture, just
      with a learned tokenizer instead of BPE. Matches how FSQ/LFQ/BSQ papers
      actually train downstream priors (frozen tokenizer, then a separate
      categorical model) — see the literature-contrast discussion earlier.
  - **FSQ attempt**: pretrain plateaued at 68.2% recon_acc (8000 steps, short
    of the 95% target). Vocab: 20,525 distinct codes / 112,500 train chunks.
    LM phase **overfit hard**: train `token_acc` →97%+, `bpb`→0.05, while
    val `bpb` climbed 3.16→4.24 over the same steps — memorizing the small
    effective training sequence, not generalizing.
  - **BSQ attempt, 5x pretrain budget (40000 steps)**: plateaued around
    77-85% train / 77-78% val recon_acc — better than FSQ's 68%, but a
    genuine plateau (train stopped climbing, oscillating rather than still
    rising), not still-improving-given-more-steps.
  - **Bug found and fixed**: `ChunkDecoder`'s docstring claimed masked
    positions could condition on unmasked ones (the entire point of
    MaskGIT), but the actual `forward()` applied a plain per-position
    `Linear`/`GELU`/`Linear` MLP to `[N, K, d_byte]` — since `nn.Linear`
    broadcasts over all but the last dimension, **every position was
    transformed completely independently**. Zero cross-position mixing,
    despite the docstring. This likely explains why the MaskGIT decoder
    change (above) only modestly improved `recon_acc` (13% vs. 9%) instead
    of solving the bottleneck outright — the mechanism MaskGIT depends on
    was never actually wired in. **Fixed**: added a real non-causal
    self-attention block (`FullSelfAttention`, full attention over
    the K positions — cheap, K is tiny) + pre-norm LayerNorms (the old
    version had none) to `ChunkDecoder`; also added a LayerNorm to
    `ChunkEncoder` (which *did* already mix all K positions correctly, via
    flatten+Linear, so didn't need attention — just had no normalization
    anywhere either). Also fixed `pos_emb`'s init from all-zeros to
    `std=0.02` (matching the rest of the codebase's init convention).
  - **Rerun with the fix, BSQ, 40000-step pretrain budget**: early signal
    is clearly better — 36% val recon_acc at step 498 vs. the pre-fix run's
    ~27% at a similar point. Full result pending (in progress).
- [ ] Sub-phase 2a (softmax-only LM) is what's implemented; sub-phase 2b
      (A-native vs. A-grounded comparison) not done — only Option A exists.
- [ ] Reconstruction accuracy vs. the old Phase-1 autoencoder not compared;
      the non-streaming encoder/decoder may reconstruct worse due to the
      chunk-boundary problem the doc warns about (§1.3).

## Phase 3 — geometric mixers

Not started.

## Experiment infrastructure (orthogonal to the phase plan)

Built out alongside the Phase 0/2 runs above, across all three training
modules — see `docs/architecture.md` for details:

- [x] `--config <file.py>` (see `configs/`), CLI flags override config values.
- [x] `--run_name` (else derived from `--config`/preset): per-run directories,
      `logs/<run_name>/run.{log,jsonl}` + `checkpoints/<run_name>/{best,last}.pt`
      — everything about one run findable by name alone.
- [x] Checkpointing: best + last, keyed to the run directory (gitignored).
- [x] `--eval_only --checkpoint_path ...`: evaluate a saved checkpoint without training.
- [x] `--qual_gen_bytes` (bytelm/qcutelm only): qualitative generation with
      ground-truth comparison and bpb-on-ground-truth, sourced from
      train/val data or user text.
- [x] Dual logging (raw text + JSONL) with elapsed-time tracking; the text
      log's interval lines also embed a `str(pbar)` snapshot (position, rate,
      postfix metrics). `logs/` gitignored.
- [x] `scripts/plot_run.py`: train/val bpb PNG from a run's `run.jsonl`,
      bigger val markers (val is logged far less often than train), and
      detects/drops stale segments from restarted runs sharing a `run_name`.
- [x] Same linear-warmup-then-constant LR schedule (`lr_at`) in all three scripts.
- [x] `qcute/bpelm.py` + `scripts/train_bpe.py`: BPE baseline, exact
      byte-weighted BPB accounting over a verified-lossless tokenizer (not a
      naive average-bytes-per-token approximation, and not exact-in-theory-
      only — the lossless roundtrip is actually checked at train time).
- [x] `scripts/diagnose_qcutelm.py`: per-position (byte 0..K-1) recon_acc
      breakdown + per-loss-term gradient norm, for any BSQ checkpoint. The
      tool that actually found the LFQ gradient-scale-explosion root cause.
- [x] `scripts/qualitative_compare.py`: same raw-byte prompts (train + val
      regions) into bytelm/bpelm/qcutelm, saves prompt/generated/ground-
      truth + bpb-on-ground-truth to `logs/qualitative_compare/`. Found the
      qcutelm generation-collapse issue that bpb numbers alone didn't show.
- [x] `qcutelm`-only training-loop flags (all BSQ-only, all in `main()`,
      not the model — see architecture.md): `--lfq`, `--uncertainty_weighting`,
      `--entropy_reg_weight`, `--bsq_sample_generation`,
      `--pretrain_ae`/`--init_encoder_decoder_from`. (`--maskgit_T` removed —
      see the decoder redesign below.)
- [ ] Not yet done: extracting the model-agnostic pieces (`Logger`,
      `Checkpointer`, `load_config_module`, `load_enwik8`, `split_train_val`,
      `lr_at`, RoPE math) into a shared `qcute/utils.py` — identified as safe
      to share (see module docstrings) but deferred as a separate decision.

## Session update — dataset standardization, decoder redesign, gradient diagnostics, `qcutelm_vlt` fork

**Dataset standardization**: `enwik8_tiny.gz` (500K bytes) removed entirely,
replaced by `enwik8_1M.gz` (1,000,000-byte prefix via `scripts/prepare_data.py
--subset_bytes`) as the one standard corpus — all three modules now default
`--data` to it with no `--n_bytes` cutoff needed. Configs renamed to drop
the premature `_converged` suffix (`bytelm_xs_mtp4.py`, `bpelm_8192.py`,
`qcutelm_bsq_k4.py`/`qcutelm_bsq_k4_frozen_vocab.py`); BPE tokenizer
retrained on the new corpus (`bpe_enwik8_1M_8192.{model,vocab}`, verified
lossless). AdamW (`betas=(0.9,0.95)`, `weight_decay=0.1`) and an optional
shared warmup→constant→cosine LR schedule (`--cosine_decay`) added to
`bytelm.py`/`bpelm.py`, matching `qcutelm.py`'s existing options — kept
off by default there (plain warmup→constant, unchanged from before)
so baseline numbers don't shift silently.

**`ChunkDecoder` architecture change — code-only, MaskGIT-style masking
removed entirely.** The decoder no longer takes a masked-byte-level input
at all: `forward(z)` (was `forward(z, x_masked)`), one-shot over all K
positions, no iterative refinement (`maskgit_decode` → `decode_bytes`,
trivial `decoder(z).argmax(-1)`). Root cause: under `full_mask` (every
position masked, the "hardest case" tried mid-session), the decoder's only
real input was `z_proj(z)` broadcast identically to every position — the
masked-byte channel contributed nothing chunk-specific — so the two-input
design had degenerated into "codes only" already, just with dead weight
(the `byte_emb`/`mask_id` machinery) still attached. Matches MaskGIT's own
two-stage assumption (Chang et al. 2022): the tokenizer is trained to
convergence and *frozen* before any masked-token scheme is introduced;
MaskGIT never masks inside tokenizer training, which is what this file had
drifted into. If masking is wanted at all now, it belongs on the
*encoder's* raw byte input (`--pretrain_encoder_mask`, `mask_bytes()` —
denoising-autoencoder-style robustness, decoder still grades against true
uncorrupted bytes), not the decoder's.

**New `ChunkEncoder`/`ChunkDecoder` capabilities** (all backward-compatible,
default = old behavior):
- `Config.context_len` (default 0) + `gather_left_context()`: asymmetric
  left-side byte context for the encoder — up to K extra bytes from the
  *previous* chunk's tail, still causal. `encode_chunks()` is the single
  wrapper every encoder call site goes through.
- `Config.encoder_layers`/`decoder_layers`/`encoder_mixer`/`decoder_mixer`
  (all `None` by default, falling back to the shared `tokenizer_layers`/
  `mixer`): per-component depth and mixer-type overrides, for asymmetric
  designs (e.g. shallow-conv encoder, deep-attention decoder).
  `encoder_layers=0` is valid — no `MixerBlock`s at all, pure flatten →
  `Linear` → bottleneck.
- `Config.quant_grad_scale` (default 1.0, no-op) + `GradScale`: QAT-style
  backward-only gradient rescale at the encoder's BSQ quantization
  boundary, to counteract two compounding attenuation sources found by a
  side gradient-norm diagnostic (below) — `sqrt(dq)` cancels the STE's
  explicit `1/sqrt(dq)` divisor exactly. Implemented, wired via
  `--quant_grad_scale`, not yet tested at a nonzero value in a real run.
- `--pq_groups` (int, default 1): replaces the old `cfg.lm_factorized_input`
  boolean for selecting `--freeze_after_pretrain`'s LM path — `1` (special
  case) = single empirical-vocab embedding table + softmax
  (`train_vocab_lm`); `!=1` = PQ/factorized per-dim embedding + per-dim BCE
  (`train_factorized_lm`, `encode_to_codes`, no vocab/UNK).
- `init_decoder_bias_to_unigram()`: initializes `ChunkDecoder.head`'s bias
  to the training corpus's log unigram byte frequency (same idea as GPT-2's
  embedding init) — a free head start, initial loss starts near the
  unigram floor (~3.1-3.5 nats) instead of the uniform floor (~5.5 nats).
  Now called unconditionally at the start of `pretrain_autoencoder()`.
- `--pretrain_lbfgs` (+ `_pretrain_autoencoder_lbfgs`, `--pretrain_lbfgs_chunks`/
  `_max_iter`/`_history`/`_no_line_search`): full-batch L-BFGS alternative
  to `pretrain_ae`'s default AdamW+random-minibatch loop, to test whether
  the full-corpus-scale plateau was optimizer noise. Result: `strong_wolfe`
  line search stalled hard (bit-identical loss for 14+ steps — BSQ's STE
  likely makes the true loss piecewise-constant once bit-signs stabilize,
  violating the line search's smoothness assumption); with the line search
  dropped, L-BFGS never plateaued in the steps tried (`train_recon_acc`
  climbing past 50%) but was too slow wall-clock for how little of the
  corpus its fixed sample covered. Reverted to AdamW; kept in the code as
  an available option.

**Full-corpus `pretrain_ae` experiment grid** (all on the standardized
900K-byte train split, `dq=18`): systematically varied tokenizer depth
(1/2/4/8 layers, symmetric and asymmetric encoder/decoder splits down to
a literal `encoder_layers=0` linear-only encoder), width (halved through
10x — a literal 10x-wider attempt overshot corpus-bit parity by ~22x,
since `out_proj`'s `seq_len*d_byte -> d_enc` matrix and the conv+MLP scale
faster than linearly with width; back-solved instead for widths landing
near 1.0x corpus-bit parity), mixer type (conv vs. attention, including
asymmetric encoder=conv/decoder=attention combinations), LR (1e-4 to 1e-3,
plain and cosine-decayed), weight decay (0.1 down to 1e-5), batch size,
and bottleneck variant (BSQ vs. LFQ — LFQ was clearly slower at matched
step counts, dropped). **Best result: 83.03%** final train `recon_acc`
(`qcutelm_bsq_k4_pq_asym_conv1_bigdec_hilr`: 1-layer conv+MLP encoder,
4-layer attention decoder at 2x width, `pretrain_lr=1e-3`, bias-init,
cosine decay) — closest any `qcutelm.py` configuration got to the 95%
target this session, still short of it.

**Side gradient-norm diagnostic** (fresh init, same architecture, CPU, run
alongside live training to avoid MPS contention): encoder gradient norms
consistently **10-28x smaller** than decoder gradient norms throughout
training, across every depth/width variant tried — including cases where
*shrinking* the encoder made the ratio *worse* (12x→28x for a linear-only
encoder vs. 8x→23x for a 1-conv-layer encoder), confirming this isn't
fixable by reallocating capacity between encoder and decoder. Root cause:
BSQ's STE quantization boundary (`bsq_quantize`) attenuates gradient via
two compounding sources — the explicit `1/sqrt(dq)` rescale, and
`F.normalize`'s own contracting Jacobian — both between the encoder and
its loss. `quant_grad_scale`/`GradScale` (above) was built to address this
directly but not yet tested. Separately, the decoder's own per-layer
gradient norms decay ~5-8x from the first to the last of 8 stacked layers
— a real, if modest, vanishing-gradient signature suggesting diminishing
returns from decoder depth past some point, independent of the encoder
issue. **No representation collapse found** at any point (0/18 dead BSQ
dimensions, healthy per-dim `z_hat` variance) — rules out codebook
collapse as an explanation for the plateau.

**`K=1` sanity check passed**: `dq=18` bits to encode 1 byte (8 bits) is
wildly over-provisioned; a healthy pipeline should trivially reach ~100%.
It did — 95.02% at step 800/2000 — confirming the repeated sub-95% results
at `K=4` are a genuine architecture/capacity-at-scale question, not a
broken pipeline (data loading, STE math, loss/target wiring all fine).

**`dq=18` combinatorial-capacity estimate** (empirical n-gram counts on
the actual 900K-byte train split): 195 distinct 1-grams, 5,620 distinct
2-grams, 30,996 distinct 3-grams, 89,225 distinct 4-grams. `dq=18`'s
codespace (`2^18=262,144`) has ~2.9x headroom over the largest single
level (4-grams) and ~2.1x headroom even under the pessimistic assumption
that all four levels need simultaneously-disjoint codes (they don't
strictly need to — decode-time chunk length is provided externally, not
inferred from the code, so the same code value can be reused validly
across different lengths). Capacity by this measure is not the binding
constraint on the plateau.

### `qcute/qcutelm_vlt.py` — variable-length causal tokenizer (new module)

A structural fork of `qcutelm.py`'s tokenizer, self-contained per the
repo's no-shared-imports convention. Encoder = causal-transformer
"pooler": `byte_emb` (no positional embedding — NoPE) → N causal
self-attention+MLP layers over 1..K bytes → take *only* the last
timestep's hidden state (a causal running summary of the whole prefix) →
project → BSQ quantize. This makes chunk length a training-time choice
(1..K), not a fixed architectural constant. Decoder = causal-transformer
"unpooler", *not* autoregressive: the code is projected and broadcast
*identically* to T timesteps, then N causal layers, then a per-position
vocab head — one forward pass, T implicit in how many positions get
broadcast to.

**Zero-KV attention is load-bearing for the decoder, not just a generic
regularizer** — verified both by derivation and empirically. Every
decoder input position gets the identical broadcast `z`, so plain causal
attention over identical, identically-scored keys produces a uniform-
weighted average — i.e. *identical* output at every position, no matter
how many layers. Concatenating a zero key/value pair before SDPA (Miller
2023, "Attention Is Off By One") fixes this: the zero key always scores 0,
contributing a fixed `exp(0)=1` term to the softmax denominator regardless
of query, so each position's real-key weight becomes a genuine function of
`n_t` (= how many real keys are causally visible at position t, which
differs by position) — a scalar per-position signal the post-attention
MLP's nonlinearity then expands into richer differentiation. Sanity check
(fresh init, 4 layers each side): all pairwise-timestep logit differences
at T=4 were substantial (0.05-0.44 max abs diff) and followed the
predicted pattern (differences shrink as position-pairs get farther apart
in causal-visible-count terms). Without zero-KV this design would be
positionally blind in its first layer regardless of depth — this repo's
earlier NoPE-only design (`qcutelm.py`) never needed this because its
decoder always had a real, position-varying masked-byte input to
differentiate on.

Curriculum (`train_curriculum()`): starts T=1, advances by 1 once the
current stage's accuracy clears `--curriculum_target_acc` (95%, gated on a
**point estimate** — the current step's own training batch, not a
multi-batch average, since the curriculum advances one way or another
regardless and the extra eval pass wasn't worth it) or
`--curriculum_max_steps_per_stage` fires as a safety-net fallback: up to
`cfg.K`. Early-stops once the final stage itself clears target, instead of
always running to `--steps`. `--replay_frac` (default 0.2): once past T=1,
that fraction of steps sample a random `T'<=T` instead of the current
stage's T, rehearsing earlier stages against catastrophic forgetting
(shared weights serve every T, nothing else stops later stages from
overwriting earlier ones) — **forced off on eval-check steps specifically**,
after a real bug where a lucky T'=1 replay batch scoring 100% falsely
triggered "target reached" at T=4 while true T=4 accuracy was only ~51%;
gating must always read a batch actually at the current stage's T.

First experiments (`configs/qcutelm_vlt_4layer.py`, 4-layer encoder+decoder,
width tuned toward corpus-bit parity — d_model=44 was ~0.95x, underfit at
T=4 within 20K steps; d_model=64 (~1.93x) with a longer 100K-step budget,
`lr_peak=1e-3`, and the fixed point-estimate/no-replay-on-eval gating
(`configs/qcutelm_vlt_4layer_nsink4.py`, `n_sink=4`, `lr_peak=6e-4`)
plateaued at 71-79% val_acc for the final ~32K of its 87K steps, never
reaching 95% at T=4 — manually stopped. Log analysis (explicit request:
"analyze the log what are failure and bad things") found two separate
problems, not one: (a) infrastructure — 5 throughput stalls of ~950-1050s
each, >50% of the run's 145-minute wall-clock was stalled, not training
(same unresolved MPS slowdown flagged elsewhere in this doc); (b) real,
sustained forgetting despite `replay_frac=0.2` — `T1_acc` held 92-99%
throughout, but `T2_acc` settled 84-91% (down from its own ~95%+ peak) and
`T3_acc` settled 68-72% (down from its own ~95%+ peak). Root-caused to the
zero-KV-derived position signal being weak and still-being-learned versus
`qcutelm.py`'s free, always-correct `pos_emb` prior — motivated the
`qcutelm_vlt2.py` fork below.

### `qcute/qcutelm_vlt2.py` — code-prefix decoder fork

Fork of `qcutelm_vlt.py` changing only the decoder's input construction.
Encoder (`PoolerEncoder`) is unchanged (still NoPE, still causal pooler).
`UnpoolerDecoder` no longer broadcasts an identical `z` to every position:
position 0's input is *always* `z_proj(z)` alone (the code, never combined
with a position embedding); positions 1..T-1's input are *always*
`pos_emb[t-1]` alone (a real `[K-1, d_model]` trainable table, never
combined with `z`) — a genuinely heterogeneous input sequence, so real
per-position differentiation exists from the first layer regardless of
zero-KV. Zero-KV attention is kept but fixed at 1 sink slot and now plays
only an "escape hatch" role (Miller 2023), not the load-bearing role it
had in `qcutelm_vlt.py`.

`configs/qcutelm_vlt2_4layer.py` (same scale as the best `qcutelm_vlt`
config, for a like-for-like comparison) reached T=4 fast but plateaued
~65-70% val_acc — gradient-norm diagnostics (side CPU script, 30 batches
on the live checkpoint) showed no pathology (per-module grad-norm CV
0.11-0.24 everywhere, no exploding/vanishing gradients) and a tiny-subset
overfit check (900 train bytes) converged cleanly to 95.78% val_acc in
7.3s — confirming the architecture itself is not broken, the plateau is a
capacity/generalization ceiling at that width on the full 900K-byte
corpus, not an architecture defect. A capacity probe at `d_model=256`
(`configs/qcutelm_vlt2_4layer_w256.py`, ~32x corpus-bit parity, deliberate
overshoot, `--no_curriculum` jump-straight-to-T=4) plus `weight_decay=0.1`
did eventually clear 95% on a 10K-byte subset (`_w256_10k.py`) by step
10,600 of a 20K-step budget, after an earlier 4K-step attempt at the same
settings had stalled ~65-70% — i.e. this scale isn't fundamentally stuck,
just slow to converge under `no_curriculum` + high weight decay; not yet
re-tried on the full corpus.

`--no_curriculum` (jump straight to `T=K`, skip staged training) added to
`qcutelm_vlt2.py`'s CLI; the eval-step forgetting-check (looping
`eval_recon_acc` over every earlier stage) is skipped entirely when this
flag is set, since there are no earlier stages to have forgotten.

### `qcute/qcutelm_vlt3.py` — single shared-weight AR tokenizer fork

Collapses the two-tower encoder/decoder split (`qcutelm_vlt`/`vlt2`'s
separate `PoolerEncoder`/`UnpoolerDecoder` weight sets) into **one**
shared-weight causal transformer (`SharedARTokenizer`) playing both roles,
like a tiny byte-level LM with a discrete bottleneck spliced into the
middle of its sequence:

- **Encoder stage**: bytes[0..T-1] + a trainable "code-query" token
  (`code_query_emb`, appended at position T) run through the shared
  `Block` stack (NoPE, causal, zero-KV escape hatch). Also gets an
  auxiliary next-token-prediction (NTP) loss on the byte positions
  themselves (standard causal LM loss, `ntp_loss_weight` default 0.5) —
  free extra gradient signal into the shared weights, bytelm.py's role but
  sharing the tokenizer's own weights instead of a separate model.
- The code-query position's final hidden state → `code_proj` → quantize
  (BSQ/FSQ/iFSQ, see below) → `z_hat`.
- **Decode stage**: `z_proj(z_hat)` becomes a *content-dependent* BOS
  token at position 0 (not a fixed learned vector), followed by
  teacher-forced byte embeddings chunk[0..T-2] at positions 1..T-1. Run
  through the **same shared weights** to predict chunk[0..T-1].

Two sequential forward calls through one `Block` stack (not a single
masked pass) — the decode stage's first token embedding needs the fully-
computed code, which isn't available until the entire encoder stage (all
layers) finishes, so full single-pass fusion isn't possible regardless of
masking; two calls avoids a fragile custom-mask implementation for an
equivalent result. Con (by design, not a bug): reconstruction decode is
genuinely autoregressive at inference (`decode_greedy`, one byte per
step) — acceptable since `T <= K` is small. Training itself is NOT
autoregressive-slow: only 2 sequential passes per step (teacher forcing
makes the decode stage a single parallel pass), not `T`.

`configs/qcutelm_vlt3_4layer.py` (`d_model=128, n_layers=4`, matching the
prior variants' scale) is, as of this writing, **the first tokenizer
variant across this entire line of experiments to actually approach the
target** — T=4 val_acc climbed steadily and monotonically (windowed-mean
check across the whole T=4 stage: 70.3% at step ~25K → 78.6% at step
~73K, no window ever lower than its predecessor, train/val gap steady at
2-4pp throughout — genuine generalization, not memorization, and not yet
plateaued when checked). Forgetting is markedly milder than `qcutelm_vlt2`
at matched scale: `T2_acc` held ~90-93% (vs vlt2's ~73-80%), `T3_acc` held
~82-85% (vs vlt2's ~55-60%). One important caveat: the run's own
"target reached" trigger at step 86,900 (`train_recon_acc=96.88%`) was a
**false positive** — same class of bug as the `qcutelm_vlt.py` replay
issue, but different mechanism: the early-stop/curriculum-advance gate
reads a single training batch's point-estimate accuracy, which is noisy
enough (64 byte predictions per batch) to spuriously cross 95% while true
val_acc was still only ~80%. Not yet fixed (candidate fix: gate on a
multi-batch average like the reverted approach in `qcutelm_vlt.py`, or
just extend `--steps` and accept the point-estimate risk). Best checkpoint
(`checkpoints/qcutelm_vlt3_4layer/best.pt`) sits at ~81% val_acc at T=4 —
the strongest tokenizer result across every architecture tried so far,
though still short of the 95% target.

Gradient-level comparison against `bytelm.py` (which reaches its best
val_bpb of 2.4872 in ~1000-1200 steps of a 2000-step run): `qcutelm_vlt3`
needed ~78K+ steps to even approach a comparable quality bar. A live
gradient-norm check on the running checkpoint (step 78,200) found the same
BSQ-boundary attenuation pattern as `qcutelm.py`/`qcutelm_vlt.py`:
`code_proj` (right before quantization) gets ~15x weaker gradient than
`head` and ~2.7x weaker than `z_proj` (right after quantization) — BSQ's
`F.normalize` + `sign()`-STE genuinely throttles the part of the network
that decides what the code says. This is a real, measured contributor to
slow convergence, compounding with (not the sole cause of) a strictly
harder objective than bytelm's (zero cross-chunk context vs. up to 256
bytes of causal context) and the inherent piecewise-loss-surface cost of
any discrete bottleneck.

#### Quantizer comparison: BSQ vs. FSQ vs. iFSQ

Added `quant_type: "bsq" | "fsq" | "ifsq"` to `qcutelm_vlt3.py`'s
`Config`/CLI (`--quant_type`, `--fsq_levels`). `SharedARTokenizer.quantize()`
is the single dispatch point; `encode()`/`encode_joint()` both call it.

- **FSQ** (`fsq_quantize`, Mentzer et al. 2023): each of the `dq`
  dimensions is quantized *independently* to one of `fsq_levels` values —
  bound to `(-1, 1)` via `tanh`, scale to the level grid, round with a
  straight-through estimator, rescale back. No shared-across-dims
  normalize step, unlike BSQ.
- **iFSQ** (user-supplied variant): identical to FSQ except the bounding
  function is `2*sigmoid(1.6*z) - 1` instead of `tanh`, intended to spread
  the pre-rounding distribution more uniformly across `(-1, 1)`.
- **BSQ**: `dq` dims, each binary (sign), plus a single L2-normalize
  applied across the whole `dq`-dim vector before the sign-STE.

Bit-budget parity for the comparison: BSQ `dq=18` → `2^18 = 262144`
codespace; FSQ/iFSQ `dq=6, fsq_levels=8` → `8^6 = 262144` — exact match.

**Empirical gradient check (fresh random init, 30 batches, `code_proj` and
`z_proj` weight-gradient norms):**

```
bsq   dq=18  code_proj grad mean=0.0961   z_proj grad mean=0.1934   (z_proj is 2.0x code_proj)
fsq   dq=6   code_proj grad mean=0.4274   z_proj grad mean=0.1104   (code_proj is 3.9x z_proj)
ifsq  dq=6   code_proj grad mean=0.3802   z_proj grad mean=0.1014   (code_proj is 3.7x z_proj)
```

FSQ's `code_proj` gradient — the exact layer BSQ was shown to starve — is
**~4.4x larger** than BSQ's at matched bit budget, and the imbalance
direction flips entirely: for BSQ, `code_proj` is the weak point relative
to what's downstream of it; for FSQ/iFSQ, it's the *strongest* point in
the encoder-side gradient chain. Matches the theoretical expectation: BSQ's
`F.normalize` is a shared, whole-vector Jacobian contracting gradient
magnitude across all `dq` dims at once (on top of the `sign()` STE); FSQ
has no such global step, just independent per-dim bounding + round-STE.

**FSQ vs. iFSQ specifically** — not distinguishable at random init (iFSQ
is marginally *lower*-gradient there, since `tanh`'s slope at `x=0` is 1.0
vs. `2*sigmoid(1.6x)-1`'s 0.8). The two bounding functions' derivatives
diverge sharply away from zero, though — this is where iFSQ's motivation
("more uniform distribution") should actually pay off, as training pushes
`code_proj`'s raw outputs toward larger magnitudes (i.e. toward decisive,
saturated quantization bins):

```
|x|    dtanh/dx   d(2*sigmoid(1.6x)-1)/dx   ratio
1.0     0.4200         0.4472                1.06x
2.0     0.0707         0.1204                1.70x
3.0     0.0099         0.0259                2.62x
4.0     0.0013         0.0053                4.08x
```

`tanh` saturates (gradient → 0) far faster than the sigmoid-based bound as
activations grow — a later-training effect, not visible at init. Ablation
plan: FSQ run first (`configs/qcutelm_vlt3_fsq8.py`, same architecture/
schedule as `qcutelm_vlt3_4layer.py` otherwise), then iFSQ at matched
settings, to see whether iFSQ's gradient advantage actually shows up once
`code_proj` activations grow past the near-zero regime where the two
bounding functions are nearly identical.

#### Reproducing

```bash
# BSQ baseline (curriculum, full corpus)
uv run python -m qcute.qcutelm_vlt2 --config configs/qcutelm_vlt2_4layer.py

# no-curriculum capacity probes (qcutelm_vlt2)
uv run python -m qcute.qcutelm_vlt2 --config configs/qcutelm_vlt2_4layer_w256.py
uv run python -m qcute.qcutelm_vlt2 --config configs/qcutelm_vlt2_4layer_w256_10k.py

# single shared-weight AR tokenizer (qcutelm_vlt3), BSQ
uv run python -m qcute.qcutelm_vlt3 --config configs/qcutelm_vlt3_4layer.py

# same, FSQ (quant_type="fsq", dq=6, fsq_levels=8 — bit-parity with BSQ dq=18)
uv run python -m qcute.qcutelm_vlt3 --config configs/qcutelm_vlt3_fsq8.py

# iFSQ ablation: clone qcutelm_vlt3_fsq8.py, set quant_type="ifsq" (not yet run)

# joint training (--joint): random init, fixed T=K throughout, no curriculum/
# replay/forgetting-check, simplified to recon_acc + aux_recon_acc (code-
# conditioned NTP via encode_joint()/forward_joint())
uv run python -m qcute.qcutelm_vlt3 --config <config> --joint

# gradient-norm / quantizer diagnostics are one-off side scripts (CPU, so
# they don't contend with a live MPS training job) — not checked into the
# repo as reusable scripts, reconstruct from this doc's snippets if needed.
```

## Session update — qcutelm_vlt4/vlt5/vlt6: strided readout, continuous
## reconstruction, and the tokenizer-as-AR-LM pivot

### `qcute/qcutelm_vlt4.py` — strided-readout tokenizer

Fork replacing `qcutelm_vlt3`'s trainable "code-query" token (which only
ever sees a bare K-byte window, no real context) with a regular,
large-context causal LM: `byte_emb` + shared `Block` stack trains with a
standard NTP loss over `context_len` bytes (128 in the first config), and
codes are read off for free at every K-th hidden state
(`h[:, K-1::K, :]`) via a small `code_net` (plain `nn.Linear` by default,
optional MLP via `code_net_layers`). Each code still reconstructs its own
K-byte block via the same code-as-BOS decode mechanism as `vlt3`, all
blocks in a context window flattened into one batched decode call.

**Sliding-window attention** (`Config.attn_window`, `-1` = full causal,
else banded width): added because a fixed `context_len` at *training*
time doesn't define a well-formed distribution for *inference* over
arbitrarily long documents — naive chunking gives boundary positions less
context than training taught them to expect, while full causal over a
whole long document gives later positions unboundedly more. A banded mask
(`mask[i,j] = 0 <= i-j < window`) reproduces, at every position of an
arbitrarily long single pass, exactly the "≤ window bytes of context"
distribution training saw. Verified: 2-layer model, `window=6`, changing
byte 0 has zero effect on hidden states at position ≥11 (receptive field
grows with depth: `(window-1)*n_layers+1`, matching exactly).

**`--joint_lm` (`forward_joint_lm`/`train_joint_lm`)**: a real latent LM
trained jointly from random init. Codes are literal indices into a fixed
vocabulary (`code_to_index`: BSQ — sum of sign-bits as binary digits,
`2^dq` vocab; FSQ/iFSQ — sum of per-dim levels as base-`fsq_levels`
digits) — no learned codebook needed, so `CodeLM` is exactly `bytelm.py`'s
recipe applied to code-space (embedding table sized to the code
vocabulary, causal `Block` stack, softmax next-code loss).
**`code_lm_detach=True`** (default): stop-gradient into the tokenizer from
`code_lm_loss`, since the code space is a moving target early in training
(tokenizer weights still shifting under `recon_loss`) — training `CodeLM`
hard against a target that won't hold still is wasted signal. First
`dq=18` attempt (`2^18=262144`-way softmax) was **catastrophically slow**
on MPS (~13-17s/step) — large-vocab softmax cost dominates; fixed by
shrinking to `dq=10` (`2^10=1024`-way, ~10 it/s). Even after the vocab
fix, `code_lm_acc` stayed near-random (1-6% vs ~0.1% chance) and
`recon_acc` flatlined for ~1400 steps — root-caused to the same
moving-target problem manifesting *despite* the detach (the code space
itself was still too unstable that early); fixed by a **`code_lm_weight`
linear warmup** (0 → target over `code_lm_warmup_steps`, default 5000),
letting the tokenizer settle into `recon_loss`-only training before
`CodeLM` training kicks in hard.

Also added (not deeply tested standalone): `--self_distill_weight` —
distills the LM stage's own full-context NTP logits into the decode
stage's (code-only) logits at the same byte positions, as an additional
soft-target signal on top of the hard reconstruction cross-entropy.

Hierarchical context (`Config.lm_context_bytes`): the tokenizer's own
attention span only needs to be small (local grounding per block) — the
*long* effective range comes from `CodeLM` stacking many codes, not from
the tokenizer's own window. `lm_context_bytes` (must be a multiple of
`context_len`) is split into independent `context_len`-byte windows
(batched together, one cheap forward pass each), all windows' codes
concatenated into one sequence for `CodeLM` — e.g. `context_len=16, K=4,
lm_context_bytes=256` → 16 independent 16-byte windows × 4 codes each =
64 codes of `CodeLM` context, reaching the full 256 bytes, while any
single tokenizer forward pass only ever attends over 16 bytes. Verified
via shape checks before use.

### `qcute/qcutelm_vlt5.py` — continuous (non-reset) reconstruction

Fork of `vlt4`'s plain (non-`joint_lm`) path: `attn_window=16` becomes
the intended default (not just an optional flag), and **the
reconstruction path is redesigned to be continuous across the whole
context instead of reset per block**. `vlt3`/`vlt4`'s `decode()` treats
every block as an independent fresh sequence (`[code, byte0, ..., byte
K-2]`, causal attention starting from nothing every time) — a real
detokenizer used by a downstream LM would instead have every previously-
decoded byte/code available as context, continuously. This fork builds
ONE long sequence spanning the whole context: for every K-byte block,
insert that block's own code right before the block starts and DROP the
block's own last byte (the byte the code was computed FROM) from the
explicit input — its information only survives implicitly via the code.
Worked example (K=4, codes c4/c8 from positions 4/8):
```
recon input:  c4, 1, 2, 3, c8, 5, 6, 7
target:        1, 2, 3, 4,  5, 6, 7, 8
```
Verified byte-for-byte against this exact example before training.
Position 5 (predicting the first byte of block 2) sees the ENTIRE
reconstructed block 1 (`c4,1,2,3`) plus its own fresh code `c8` — later
blocks get progressively more accumulated context, matching how a real
downstream LM would use this as a detokenizer.

**Per-chunk accuracy breakdown**: the pooled `val_recon_acc` mixes chunk 0
(zero accumulated context, the true hard case) with later chunks (more
context, easier) — added `chunk0_acc`/`chunk1plus_mean_acc`/
`chunk_last_acc` to `eval_model`'s logging so the hard case isn't hidden
by an average pulled up by easier chunks.

**Chunked sliding-window attention** (`ZeroKVCausalSelfAttention.
_forward_chunked`): the original windowed-mask implementation
(`attn_mask` passed to SDPA) is still `O(T^2)` — a boolean mask only
controls *which* keys count, PyTorch's SDPA still computes the full dense
`Q·Kᵀ` regardless. Real speedup requires never materializing the full
`T×T` matrix: split `T` into non-overlapping chunks of size `W=window`;
each chunk's queries only ever need this chunk's + the previous chunk's
keys/values (a window of size `W` reaching back from anywhere in chunk
`c` can reach at most into chunk `c-1`), so the needed mask is a single
fixed `[W, 2W]` pattern reused for every chunk/batch/layer. **First
implementation had a real bug**: chunk 0 has no real previous chunk (its
`kc_prev`/`vc_prev` are zero-padding), but the shared mask still marked
some prev-chunk slots "allowed" for chunk 0's queries — diluting the
softmax denominator with fake zero-key contributions (verified via a
position-by-position diff against a from-scratch manual per-position
reference: chunk 0 wrong, chunk 1+ exact). Fixed with a per-chunk mask
override (chunk 0's prev-chunk columns forced `False`). Re-verified:
bit-exact match to the dense path (`~1e-7` max diff) across `T`/`window`
combos, gradients also verified matching (`~1.8e-7` max diff). **Real
speedup was modest, not dramatic** (`context_len=512`: 1.32 it/s chunked
vs 1.10 it/s dense, ~20%, not the ~16x pure-attention-FLOP math implied)
— at this model scale (`d_model=128, mlp_mult=4`) the MLP/linear layers
dominate total compute and scale with `T` regardless of windowing;
attention's `O(T²)→O(T·window)` reduction only helps attention's own
(smaller) share of the cost, and the chunked path's extra small-op count
(permutes/cats/reshapes) has real MPS kernel-launch overhead too.

**`CodeLM` plugged in, not detached** (per explicit design: "no
supervision on code, just pure ntp on tokenizer pre codelm level and the
recon with code post codelm... pretraining is basically special case
where codelm=identity"): a causal transformer sitting between code
production and decode, operating on the CONTINUOUS code embedding
sequence (`z_proj(z_hat)`, not `code_to_index` — avoiding both the
moving-target and huge-vocab-softmax problems from `vlt4`'s `--joint_lm`
entirely, since there's no separate code-level classification loss at
all). Residual with a **zero-initialized output projection**:
`forward(x) == x` exactly at initialization — verified bit-exact against
`use_code_lm=False` (identical loss to 15 decimal places at init) —
literally realizing "pretraining is the `codelm=identity` special case"
as the model's actual starting point, not a separate mode. Gradient does
flow through (verified: `code_net`, upstream of `code_lm`, gets nonzero
gradient from `recon_loss`); the inner `code_lm.blocks` show zero
gradient on the very first step specifically because `out_proj` is
zero-initialized (its own Jacobian is zero at step 0) — expected zero-
init-residual behavior, not a bug — `out_proj`'s own weights get gradient
first, blocks start receiving gradient from step 2 onward once `out_proj`
moves off zero.

### `qcute/qcutelm_vlt6.py` — the tokenizer IS an AR latent LM

Deliberate departure from every earlier variant, which all trained the
tokenizer as an **autoencoder** (code computed from block i's own bytes,
used to reconstruct those SAME bytes). Pipeline: `bytes → encoder → code
→ codelm → codepred (factorized) → next code → decoder → bytes`. **Only
one loss**: byte-level NTP cross-entropy at the decoder's output — no
reconstruction loss, no auxiliary encoder-side NTP loss, no code-level
classification loss. `codelm` processes the sequence of TRUE codes
causally (teacher-forced, like a normal LM); `codepred` predicts a soft,
differentiable representation of the NEXT code — factorized sigmoids
(BSQ/LFQ: `dq` independent bits, `2*sigmoid(logits)-1`) or factorized
softmax (FSQ/iFSQ: `dq` independent per-dimension categoricals over
`fsq_levels`, soft expected value) — matching `bsq_quantize`'s/
`fsq_quantize`'s own value ranges so predictions are in-distribution for
the decoder. The decoder then reconstructs a block **it was never
shown** — genuine held-out next-block prediction, not autoencoding.
`codepred` has no loss of its own; all its gradient comes indirectly
through the decoder's byte NTP loss backpropping through the whole chain.

**No-leakage verified** before any training run: perturbing a byte inside
block 2 changes block 2's own code but leaves `pred_soft[:,1]` (which
predicts block 2 from blocks 0-1 only) exactly unchanged; once block 2's
code does change, `pred_soft[:,2]` (predicting block 3, which legitimately
depends on block 2's code) changes too. Causal chain confirmed correct.

**`quant_type="none"` ablation**: `quantize()` becomes identity
passthrough (fully continuous code), `codepred`'s head becomes a plain
unbounded linear prediction (no sigmoid/softmax squashing). Diagnostic
only — isolates whether the discrete bottleneck itself costs real
convergence speed, does NOT mean continuous is a viable replacement for
the project's generative goal: `codepred` here is trained by pure
backprop with no distributional loss of its own, so it will tend toward
regression-to-the-mean rather than a genuinely samplable next-code
distribution (the same reason plain pixel-wise L2 losses give blurry
image generations instead of sharp ones) — real continuous generation
needs a diffusion/flow/mixture-density-style head, not present here.

**Generation code** (`generate`/`decode_greedy_block`/`qualitative_gen`):
genuine nested-AR sampling — block-level AR (`codelm` predicts the next
code from purely causal past codes) nested with byte-level AR
(`decode_greedy_block`, byte-by-byte greedy from the predicted code as
BOS) — no ground truth used anywhere in `generate()`. `qualitative_gen`
logs prompt/generated/ground-truth at a **fixed** seed offset
(reproducible across evals) every `--gen_every` steps (default 1000),
with byte-match% against the real continuation.

**First real training run** (`configs/qcutelm_vlt6_ifsq_vs_bpelm.py`,
iFSQ, then several capacity/FLOP-matching variants, 8000-step budget = 4x
`bytelm_xs_mtp4`'s 2000-step budget): `val_bpb`/`val_next_block_acc`
improved steadily and monotonically for the whole observed run (no
plateau, e.g. 3.90→3.08 bpb and 27%→40% acc from step ~400 to ~3900) —
genuine learning on a strictly harder task than plain NTP (predicting an
entire unseen K-byte block from causally-derived codes only, not just the
next byte with full left context). **Greedy-decoded qualitative samples
lag the underlying metrics badly** — repeatedly collapsed into
repetition loops ("the the the...", later "the and the and...") even as
`next_block_acc` climbed past 40%; occasional breakouts produced
topically-relevant, syntactically real fragments (valid `[[Article|
display]]` MediaWiki link syntax, real word fragments like "Atlas
Shrugged", "Austria", "Algeria" matching the African-countries prompt
context) — a known greedy-decoding artifact (locally-highest-probability
tokens forming self-reinforcing loops on an undertrained model), not
necessarily reflecting true model quality; the teacher-forced
val_next_block_acc is the more reliable signal to track.

**Budget/comparability notes**: `bytelm_xs_mtp4`/`bpelm_8192` are 2000-
step, single-forward-pass, ~5-6 minute baselines; `qcutelm_vlt6` inherited
`vlt3`-`vlt5`'s 100K-step convention by default (not matched) and costs 3
forward passes/step (encoder + decode + codelm) at `context_len` up to
1024, so per-step cost and total budget both need explicit matching for a
fair comparison — see the `Reproducing` section below for exactly which
configs match which baseline and how. `bytelm_xs` uses `mtp_heads=4`
(structurally predicts 4 future bytes per position already, though only
head-0's bpb is the reported/comparable metric) — a real architectural
parallel to `vlt6`'s own K=4 grouping worth keeping in mind for any
bytelm-specific comparison, though it doesn't apply to bpelm (no MTP).

**Context-length matching is NOT symmetric between the two baselines**:
`bytelm`'s `context=256` means 256 raw bytes (1:1 token↔byte). `bpelm`'s
`context=256` means 256 BPE **tokens** — per its own "matched-bandwidth
(~4 bytes/timestep)" convention, that's ~1024 raw bytes, not 256.
`qcutelm_vlt6`'s codes are structurally bpelm's-tokens' analogue (each
compresses K bytes), not bytelm's raw-byte analogue — so matching bpelm
correctly means `CodeLM`'s own context (in code-units) = 256 directly
(`context_len=1024` at `K=4`), NOT `context_len/K` derived from bytelm's
256-byte span (that derivation — `context_len=256` → 64 codes — is what
matches *bytelm* instead, a different config).

**Params vs. FLOPs cannot be matched to a baseline simultaneously** at a
fixed `context_len`, because `vlt6` processes 3 passes/step (encoder +
decode + codelm) vs. bytelm/bpelm's 1 — a structural ~2-4.5x more total
token-positions/step depending on which baseline's context convention is
used, independent of `d_model`/`n_layers`. Params scale only with
`d_model²·n_layers` (sequence-length-independent); FLOPs scale with
`d_model²·n_layers·(tokens processed)`. Matching one forces a mismatch on
the other. A useful additional lever: **`attn_window` adds real attention
compute with ZERO added parameters** (it only changes how much of an
already-computed K/V each query attends to) — used to "compensate less
params" when asked for ~2x more FLOPs than a params-matched baseline:
widening `attn_window` 16→64 adds real compute for free, letting a
smaller width/depth bump reach a target FLOP level than a naive
width/depth doubling would need.

Also: since bytelm/bpelm have no *learned* tokenizer (bytelm has none at
all; bpelm's BPE tokenizer is non-parametric), `vlt6`'s encoder/decoder
is a REAL parameter cost the baselines don't pay — kept deliberately
cheap (`d_model=64, n_layers=2` → 0.133-0.274M params, <1M) so most of the
capacity budget goes to `CodeLM` (the actual "language model" component,
sized closer to the baselines' own ~3.7M).

#### Reproducing

```bash
# qcutelm_vlt4 — strided-readout tokenizer, plain (no joint LM)
uv run python -m qcute.qcutelm_vlt4 --config configs/qcutelm_vlt4_ctx128.py

# qcutelm_vlt4 --joint_lm — real latent LM, code_lm_weight warmup
uv run python -m qcute.qcutelm_vlt4 --config configs/qcutelm_vlt4_joint_lm.py

# qcutelm_vlt5 — continuous non-reset reconstruction, BSQ
uv run python -m qcute.qcutelm_vlt5 --config configs/qcutelm_vlt5_bsq.py

# qcutelm_vlt5 with CodeLM plugged in (use_code_lm=True, not detached)
uv run python -m qcute.qcutelm_vlt5 --config configs/qcutelm_vlt5_codelm.py

# qcutelm_vlt6 — tokenizer-as-AR-LM, single loss, matched against bytelm
# (context_len=256 -> CodeLM sees context_len/K=64 codes)
uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_ifsq_cheap_tokenizer.py

# qcutelm_vlt6 matched against bpelm instead (context_len=1024 -> 256 codes,
# matching bpelm's 256-TOKEN context directly, not bytelm's byte span)
uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_ifsq_vs_bpelm.py

# qcutelm_vlt6, FLOP-proxy exactly matched to bpelm_8192 (d^2*n_layers*tokens)
uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_ifsq_flopmatch_bpelm.py

# qcutelm_vlt6, ~2x that FLOP budget, leaning on a wider attn_window (64) to
# "compensate less params" than a naive width/depth doubling would cost
uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_ifsq_2xflops_leanparams.py

# qcutelm_vlt6, quant_type="none" ablation (no codebook, fully continuous
# code) — diagnostic only, see caveat above before treating as a real result
uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_none_2xflops.py

# qcutelm_vlt6 generation is trained-in (--gen_every, default 1000 steps) —
# no separate script; see qualitative_gen()/generate() in qcutelm_vlt6.py
```

## Session update — qcutelm_vlt6: composable losses, RoPE/zero-KV, shared/
## separate encoder-decoder, 16-cell grid

### New `qcutelm_vlt6.py` capabilities

**Composable losses** (`Config.main_ntp_weight`/`aux_recon_weight`/
`code_match_weight`, any combination active simultaneously, each only
computed when its weight is nonzero — `main_ntp` additionally has a
`force_main_ntp_metrics` escape hatch so `eval_model` can still report a
comparable `val_bpb` even when a run doesn't train on it directly):
- `main_ntp` (original): `decode(codepred(codelm(codes[:i])))` vs
  `block[i+1]`, one chunk ahead.
- `aux_recon`: `decode(code[i])` vs `block[i]` directly, zero-shift,
  bypassing `codelm` — reuses the same `decode_block` mechanism, gives
  the encoder/decoder a short direct gradient path. Trains the decoder
  when `main_ntp_weight=0` (otherwise the decoder gets zero gradient —
  `main()` warns at startup if both `main_ntp_weight` and
  `aux_recon_weight` are 0).
- `code_match`: factorized discrete-aware match between `codepred`'s raw
  logits and the encoder's own **detached** true next code — BCE-per-bit
  (BSQ/LFQ) or CE-per-dimension over `fsq_levels` (FSQ/iFSQ). No decode
  pass needed. Empirically (single-run check, `code_match`-only + `aux_recon`
  for decoder gradient): `code_match_loss` plateaus/degrades slightly on
  its own while `aux_recon_acc` climbs fine — consistent with `codelm`
  chasing a genuinely moving target (the tokenizer's own code distribution
  keeps shifting under `aux_recon`'s training), not a dead-gradient issue.
  Adding `main_ntp` back in stabilizes `code_match_loss`'s trend (rises
  early, plateaus, turns slightly down) rather than fixing it outright.

**RoPE / zero-KV as independent toggles** (`Config.use_rope`,
`Config.use_zero_kv`, `Config.rope_base`) — both default to the original
design (NoPE + zero-KV sink). `use_rope=True` applies rotary embeddings
(`rope_cos_sin`/`apply_rope`, identical math to `qcute.bpelm`, duplicated
per convention) to q/k in every attention call, threaded through
`Block`/`CodeLM`/`run_blocks`/`run_decoder_blocks` (cos/sin recomputed
fresh per call from the current sequence length — this makes `generate()`'s
growing-sequence greedy decode correct for free, no separate handling
needed). `use_zero_kv=False` drops the sink entirely; combined with
`use_rope=True` this reproduces `qcute.bpelm`'s `CausalSelfAttention`
exactly (plain `is_causal=True` SDPA, no sink) — true "bpelm parity."

**Efficient windowed attention now covers the no-sink case too**
(`_forward_chunked_no_sink`) — the original chunked `O(T·window)`
implementation (`_forward_chunked`, from the `qcutelm_vlt5` session) only
existed for the zero-KV branch; `use_zero_kv=False` + `window` set was
silently falling back to an `O(T²)` dense-masked path (a boolean
`attn_mask` doesn't reduce SDPA's actual compute — it only removes the
fast `is_causal=True` kernel path). Confirmed via direct comparison:
`context_len=1024`, `attn_window=64` — dense full-causal `~2.4 it/s`,
`O(T²)` dense-windowed also `~2.4 it/s` (no better — the window wasn't
helping at all without chunking), chunked-no-sink `~3.2 it/s`. Verified
bit-exact (`~1e-7` max diff, gradients matching) against the dense-no-sink
path across several `T`/`window` combinations before trusting it.
Mechanism (same trick as the zero-KV version): split `T` into
non-overlapping chunks of size `W`; each chunk's queries only ever need
this chunk's + the previous chunk's K/V (a window of size `W` reaching
back from anywhere in chunk `c` reaches at most into chunk `c-1`); fold
the chunk index into the batch dim and call SDPA **once**, batched, on
`[B·n_chunks, H, W, hd]` queries against `[B·n_chunks, H, 2W, hd]` K/V —
`O(T·window)` instead of `O(T²)`, still just one SDPA call. Chunk 0's
"previous chunk" is zero-padding with its mask columns forced `False`
(no real previous chunk exists) — the same fix the zero-KV version needed.

**`shared_encoder_decoder`** (`Config`, default `True` = original
behavior, verified via object-identity check that `z_proj is z_proj_dec`
and `blocks is dec_blocks` when shared — zero overhead/behavior change).
`False`: encoder and decoder get fully independent weights
(`byte_emb`/`blocks`/`ln_f`/`head`/`z_proj` all duplicated as
`dec_byte_emb`/`dec_blocks`/`dec_ln_f`/`head`/`z_proj_dec`), symmetric by
default (`dec_d_model`/`dec_n_heads`/`dec_n_layers`/`dec_mlp_mult` all
`None` → mirror the encoder's own dims exactly — genuinely separate
weights, not a smaller/larger split) with `dec_*` overrides available for
an asymmetric split later. Caught a real bug while building this: `z_proj`
was serving two different roles (feeding `CodeLM`'s input at `d_model`
width, AND building decode's BOS) — once separated, decode's BOS needs
`dec_d_model` width instead, so `z_proj` (→ `d_model`, always, feeds
`CodeLM`) and `z_proj_dec` (→ `dec_d_model`, feeds decode's BOS) had to
become genuinely separate projections.

### bytelm/bpelm comparability notes (from getting `qcutelm_vlt6` runs matched)

- `bytelm`'s `context=256` means 256 raw **bytes**. `bpelm`'s `context=256`
  means 256 BPE **tokens** — per its own "matched-bandwidth (~4
  bytes/timestep)" convention, that's ~1024 raw bytes, not 256. `vlt6`'s
  codes are structurally bpelm's-tokens' analogue (each compresses `K`
  bytes), not bytelm's raw-byte analogue — matching bpelm correctly means
  `CodeLM`'s own context (in code-units) = 256 directly (`context_len=1024`
  at `K=4`), not `context_len/K` derived from bytelm's 256-byte span
  (`context_len=256` → 64 codes matches *bytelm* instead — a different,
  also-valid config, not the same target).
- Params vs. FLOPs **cannot** be matched to a baseline simultaneously at a
  fixed `context_len`, because `vlt6` processes 3 passes/step (encoder +
  decode + codelm) vs. bytelm/bpelm's 1 — structurally more total
  token-positions/step regardless of `d_model`/`n_layers`. Solved the
  general 2-unknown linear system (`X`=encoder/decoder's `d²·n` budget,
  `Y`=CodeLM's): at `context_len=256` (bytelm-byte-matching) a valid
  positive asymmetric split exists (~43%/57%); at `context_len=1024`
  (bpelm-token-matching, what all the grid configs use) it degenerates to
  requiring a zero-capacity encoder — not realizable, confirming why
  those two targets can't both be hit at once for these specific configs.
  `attn_window` is a useful lever here since it adds real compute with
  **zero** added parameters (only changes how much already-computed K/V
  each query attends to) — used to "compensate less params" when asked
  for ~2x more FLOPs than a params-matched baseline.
- `bytelm_xs` uses `mtp_heads=4` (structurally predicts 4 future bytes per
  position already, though only head-0's bpb is the reported metric) — a
  real architectural parallel to `vlt6`'s own `K=4` grouping worth keeping
  in mind for bytelm-specific comparisons (doesn't apply to bpelm, no MTP).
- `bpelm`/`bytelm`/`qcutelm` all use RoPE (`rope`/`rotate_half` present in
  each). `vlt6`'s default (`use_rope=False`) is NoPE — a real architectural
  difference from every baseline, independent of the loss/objective
  differences, now toggleable via `use_rope=True`.
- Tokenizer overhead is a real param cost for `vlt6` that neither baseline
  pays (bytelm has no tokenizer at all; bpelm's BPE tokenizer is
  non-parametric) — kept deliberately cheap (`d_model=96, n_layers=2` →
  ~0.27-0.46M params depending on shared/separate) so most of the budget
  goes to `CodeLM` (~2.4M, closer to the baselines' own ~3.7M).

### Qualitative generation findings (across all main_ntp-trained runs so far)

Greedy-decoded samples lag the underlying teacher-forced metrics badly —
repeatedly collapse into repetition loops ("the the the...", "the and the
and...") even as `val_next_block_acc` climbs past 35-40%; occasional
breakouts produce topically-relevant, syntactically real fragments (valid
`[[Article|display]]` MediaWiki link syntax, real n-grams like "Atlas
Shrugged", country names matching the prompt's African-countries context).
Known greedy-decoding artifact (locally-highest-probability tokens forming
self-reinforcing loops on an undertrained model) — `val_bpb`/
`val_next_block_acc` (teacher-forced) are the reliable signals to track,
not the qualitative samples, at this training budget (8000 steps).

### 16-cell grid (quant_type x loss_type x shared_encoder_decoder x use_zero_kv)

Running unattended via a sequential bash driver
(`run_vlt6_grid.sh`, one config after another — respects the
one-job-at-a-time rule by construction, no orchestration needed) wrapped
in `caffeinate -i -s` (testing whether this resolves the recurring MPS
throughput-stall issue flagged earlier in this doc — stalls were
hypothesized to be OS-level idle/sleep-related power management
interfering with the MPS process). All 16 cells share: RoPE, `context_len
=1024`, `attn_window=64`, `d_model=96/n_heads=4/n_layers=2/mlp_mult=4`
(tokenizer), `lm_d_model=256/lm_n_heads=4/lm_n_layers=3/lm_mlp_mult=4`
(CodeLM), `steps=8000`, `gen_every=1000`. `dq`: ifsq=5 (`8^5=32768`
codespace), bsq=13 (`2^13=8192` — exact bpelm-vocab match, since 8192 is
a clean power of 2 unlike ifsq's levels=8 base).

| quant | loss | encdec | zerokv | config |
|---|---|---|---|---|
| ifsq | ntp | shared | False | `qcutelm_vlt6_rope_bpelm_parity.py` |
| ifsq | ntp | separate | False | `qcutelm_vlt6_separate_encdec.py` |
| ifsq | aux | shared | False | `qcutelm_vlt6_grid_ifsq_aux_shared.py` |
| ifsq | aux | separate | False | `qcutelm_vlt6_grid_ifsq_aux_encdec.py` |
| bsq | ntp | shared | False | `qcutelm_vlt6_grid_bsq_ntp_shared.py` |
| bsq | ntp | separate | False | `qcutelm_vlt6_grid_bsq_ntp_encdec.py` |
| bsq | aux | shared | False | `qcutelm_vlt6_grid_bsq_aux_shared.py` |
| bsq | aux | separate | False | `qcutelm_vlt6_grid_bsq_aux_encdec.py` |
| ifsq | ntp | shared | True | `qcutelm_vlt6_grid_ifsq_ntp_shared_zerokv.py` |
| ifsq | ntp | separate | True | `qcutelm_vlt6_grid_ifsq_ntp_encdec_zerokv.py` |
| ifsq | aux | shared | True | `qcutelm_vlt6_grid_ifsq_aux_shared_zerokv.py` |
| ifsq | aux | separate | True | `qcutelm_vlt6_grid_ifsq_aux_encdec_zerokv.py` |
| bsq | ntp | shared | True | `qcutelm_vlt6_grid_bsq_ntp_shared_zerokv.py` |
| bsq | ntp | separate | True | `qcutelm_vlt6_grid_bsq_ntp_encdec_zerokv.py` |
| bsq | aux | shared | True | `qcutelm_vlt6_grid_bsq_aux_shared_zerokv.py` |
| bsq | aux | separate | True | `qcutelm_vlt6_grid_bsq_aux_encdec_zerokv.py` |

Results TBD — this section describes the design; findings to be appended
once the grid completes (or is stopped/checked partway through).

    # driver script lives in the session's scratchpad, not checked into the
    # repo — reconstruct from the CONFIGS array above if needed:
    for cfg in configs/qcutelm_vlt6_{rope_bpelm_parity,separate_encdec}.py \
               configs/qcutelm_vlt6_grid_*.py; do
        uv run python -m qcute.qcutelm_vlt6 --config "$cfg"
    done

## Discussion — at what regime would qcute's tokenizer-as-AR-LM actually
## beat BPE, and how does it compare to audio/video/image tokenizers?

Prompted by: "at what regime and search space hypothetically this qcute
tokenizer wins over bpe, consider llm bpe can be huge like 60k or 100k
vocab, also vs audio video image tokenizer." Analytical discussion only —
no new code or runs; grounded in this session's actual measurements
(FLOP/param comparisons, quantizer instability findings) rather than
speculation, and stated with the caveat that the 16/17-cell grid hasn't
finished, so the bpb side of this comparison is still unproven.

### The honest baseline: right now, on this corpus, BPE is winning on every axis that matters

At the scale actually trained this session (enwik8_1M.gz, vocab=8192,
d=256): bpelm_8192 best val_bpb **2.3679** vs bytelm_xs_mtp4 **2.4872** —
BPE is *already* the best of the three baselines despite (a) zero runtime
tokenization cost (a lookup table, trained once via a few seconds of
frequency counting, not gradient descent) and (b) zero training
instability (compare to this session's entire BSQ-gradient-imbalance /
moving-target / quantizer-collapse saga — none of which BPE has any
analogue of). qcute's own FLOPs are only ~19% below bpelm's at matched
context (2180.86M vs 2684.35M, see the width-symmetric/FLOP-breakdown
benchmarking above) — not a dramatic margin — and no qcute variant has yet
matched bpelm's val_bpb. So the fair starting position for this discussion
is: BPE is a very strong, very cheap baseline, and any claim that qcute
wins has to identify a *specific regime* BPE is structurally bad at, not
just assert general superiority.

### Regime A — vocabulary scaling (the clearest structural edge)

BPE's cost scales with vocab size roughly as `O(vocab * d_model)` for the
embedding table plus a full `vocab`-way softmax on every output position.
At vocab=8192 (this session's scale) that's cheap; at the 60k-100k vocab
LLMs actually use in production, it stops being cheap:

- Embedding+head params alone: `100,000 * d_model` — at d_model=256 that's
  25.6M params just for the vocab table, before a single transformer layer.
  At GPT-scale d_model (2048+) it's hundreds of millions, and it scales
  *linearly* in vocab with no way to share structure across tokens.
- Every training/inference step pays a `vocab`-way softmax — real
  wallclock and memory cost (logits are `[B, T, vocab]`), independent of
  how rare most of those 100k tokens actually are (Zipfian: the tail is
  wildly undertrained per-token relative to its parameter cost).

qcute's code space is *factorized*, not monolithic: `dq` independent
per-dimension distributions (sigmoids for BSQ/LFQ, `fsq_levels`-way
softmaxes for FSQ/iFSQ) address `fsq_levels^dq` (or `2^dq`) *combinations*
using only `O(dq * fsq_levels)` output parameters and `dq` independent
small softmaxes — not one huge one. Concretely, matching BPE's
100k-token addressability needs `dq * log2(fsq_levels) >= log2(100000) ≈
17` bits — e.g. `dq=6, fsq_levels=8` (18 bits, 262,144-code space) — via
six independent 8-way softmaxes (48 output classes total) instead of one
100,000-way softmax. This is the single clearest place the structural bet
should pay off, and the gap *grows* with target vocabulary — it's an
argument that gets stronger exactly where "just use a bigger BPE vocab"
gets more expensive, not weaker.

Caveat: this session never actually tested a vocab-matched-to-100k
config — `dq=5/6` here targets bpelm_8192, not a 100k-vocab regime. This
is a real, testable prediction (build a `qcutelm_vlt6` config with
`dq=6, fsq_levels=8` or larger and compare param/FLOP counts against a
hypothetical 100k-vocab bpelm), not a confirmed result.

### Regime B — open-vocabulary / multilingual / out-of-distribution bytes

BPE's merge table is a fixed lookup, frozen after training on one
corpus/language mix. On out-of-distribution byte sequences (a script or
language underrepresented in the merge-training corpus, binary data,
adversarial input) it degrades to byte-level fallback — a well-documented
failure mode where "matched bandwidth" (bytes/token) collapses exactly
where you'd most want compression to hold up. qcute's encoder is a
learned *function* of bytes, not a table — in principle it generalizes
continuously to unseen byte patterns via interpolation in latent space,
rather than falling off a cliff into single-byte tokens. This is a
plausible advantage but **untested here**: enwik8 is a single,
homogeneous English corpus, so this session has produced zero evidence
either way — it would need a deliberately OOD eval split to check.

### Regime C — non-stationary / continually-adapting corpora

Changing BPE's vocabulary (e.g. as a model's training distribution shifts
over time) requires retraining the merge table and, in practice,
retraining the embedding/head from scratch — a hard discontinuity. qcute's
tokenizer is trained jointly with (or ahead of) the downstream LM and
could in principle keep adapting online. Also untested here (single
static corpus, no distribution-shift setup) — a real potential advantage,
not a demonstrated one.

### Regime D — long-context, attention-bound compute (this project's actual structural bet)

This is the regime `qcute` is structurally built for: CodeLM operates on
`context_len / K` tokens, not `context_len` bytes or `context_len/~4`
BPE tokens — a *second* compression factor stacked on top of whatever
BPE-equivalent compression the encoder itself achieves. This only pays
off once attention's `O(T²)` term stops being dominated by the MLP's
`O(T)` term — which, per this session's own finding, is exactly the
regime our own FLOP-counting tooling is blind to (`FlopCounterMode`
reports **zero** FLOPs for `scaled_dot_product_attention` — confirmed by
direct test — so every params/FLOPs table in this doc undercounts
attention entirely; the real wallclock signal, ~2.4→~3.1-3.2 it/s from
chunked windowing, is the only trustworthy proxy so far). At short
context (256-1024 bytes, this session's whole regime) attention is a
minor cost and this advantage barely shows up — which matches the
observed ~19% FLOPs edge over bpelm being modest, not dramatic. At long
context (10k+ tokens), a 4x-or-more token-count reduction compounds
quadratically in the attention term specifically, in a way no
BPE-vocab-size increase can replicate (BPE's compression saturates —
merge frequency is Zipfian, diminishing returns past a few hundred
thousand merges — while `K` is a free structural dial with no equivalent
diminishing-returns wall, modulo reconstruction quality per block).

### Where BPE wins outright, no hedging

- **Zero runtime tokenization cost.** BPE is a lookup; qcute pays a
  mandatory encoder+decoder forward pass every step, forever. This
  session's own numbers show that tax is real: total FLOPs land close to
  bpelm's despite a much narrower tokenizer, because CodeLM's own cost
  plus encode/decode overhead adds back most of what K-fold compression
  saves (see the FLOP-breakdown benchmarking above).
- **Zero training cost/instability.** BPE trains via frequency counting in
  seconds. qcute's entire session-long saga — BSQ's 10-28x
  `code_proj` gradient imbalance from `F.normalize`'s Jacobian
  contraction, the moving-target problem when training CodeLM against a
  shifting tokenizer, code_match's decoder-starvation failure mode if
  weighted wrong — has no BPE analogue whatsoever. The 17-cell grid this
  session is running exists *because* qcute's design surface (quantizer
  type, loss composition, encoder/decoder sharing, dq/fsq_levels) is
  large and non-obvious; BPE has no equivalent hyperparameter search.
- **Exact invertibility.** BPE's tokenize→detokenize round-trip is always
  lossless by construction. qcute's decoder is a *learned, approximate*
  inverse — reconstruction accuracy is never guaranteed, and this
  session directly observed imperfect byte-match in early generations.
- **Train/inference consistency.** BPE's merge algorithm is identical at
  train and inference time, always. qcute's codes depend on a trained
  network whose behavior on OOD input can degrade unpredictably, not just
  "more fragmented" the way BPE's fallback does.

### vs. audio/video/image tokenizers — a different comparison entirely

Text has something audio, video, and images structurally lack: a cheap,
lossless, zero-training discrete tokenizer already exists for it (BPE).
Audio/image are continuous-valued and locally dense/redundant in a way
that frequency-counting discrete merges can't exploit directly — there is
no BPE-equivalent for raw waveforms or pixels. That's precisely why
learned neural tokenizers (VQ-VAE, RVQ codecs like SoundStream/EnCodec,
FSQ-based image tokenizers) are already the *standard, undisputed*
approach in those domains, not a hypothesis to test — the alternative
there isn't "a free 5-minute counting algorithm," it's raw floats. FSQ
itself (Mentzer et al. 2023, the quantizer this session's `ifsq`/`fsq`
variants are based on) originates from that image/audio-tokenizer
literature, not from text.

The implication: qcute is attempting, for text/bytes, what VQ-VAE-style
tokenizers already do uncontested for audio/image/video — but the bar is
categorically higher for text, because BPE is a strong free alternative
that has no counterpart in those other domains. This reframes the whole
project's likely payoff surface: qcute is more likely to win decisively
on byte streams that are *not* natural-language text — compressed binary
data, serialized sensor/log data, or other domains where bytes are really
standing in for a continuous/dense signal the way audio samples or pixels
do — than to beat BPE outright on English text, unless one of regimes
A-D above is decisively in play (huge target vocabulary, OOD/multilingual
generalization, non-stationary domains, or long-context compute-bound
serving).

### Net read

Nothing here is confirmed by a run — it's a structural argument for
*where to point the next experiments*, not a result. The regimes worth
actually testing, roughly in order of how directly this session's tooling
could check them: (1) a large-`dq` config compared against a
hypothetical/real large-vocab bpelm at matched addressable-code-space —
tests Regime A directly; (2) a long-context config (well past 1024 bytes)
with wallclock (not FLOP-counter) timing — tests Regime D, and is the one
this session's own tooling under-measures; (3) an OOD or multilingual
eval split — tests Regime B, needs new data, not yet available in this
repo.

## 16-cell grid results — partial, saved before the run was stopped to
## prioritize qcutelm_vlt7

The grid (see table above) was interrupted after 8/16 non-zeroKV cells
plus 1/8 zeroKV cells (cell 9, `ifsq_ntp_shared_zerokv`, in progress) to
free the GPU for `qcutelm_vlt7`'s first real-data test. BSQ cells' results
(all 4 non-zeroKV ones completed) saved here before that queue was torn
down:

| config | train_bpb (at best) | best val_bpb | notes |
|---|---|---|---|
| `qcutelm_vlt6_grid_bsq_ntp_shared` | 2.1728 | 2.6698 | |
| **`qcutelm_vlt6_grid_bsq_ntp_encdec`** | 2.0262 | **2.5872** | best qcute result of the whole grid — separate (not shared) encoder/decoder weights |
| `qcutelm_vlt6_grid_bsq_aux_shared` | 4.5680 | 4.5923 | **not comparable** — `main_ntp_weight=0`, so `val_bpb` comes from `force_main_ntp_metrics` running an untrained decode path; noise, not signal |
| `qcutelm_vlt6_grid_bsq_aux_encdec` | 4.6250 | 4.6739 | same caveat as above |

iFSQ non-zeroKV cells (`ntp_shared` 2.6993, `separate_encdec` 2.6717,
`aux_shared`/`aux_encdec` — same aux caveat, ~4.6-4.7) are in the full
table earlier in this doc. Both baselines still ahead of every `ntp`-loss
cell: bytelm 2.4872, bpelm 2.3679 (old 2000-step run; an 8000-step
rerun — `bpelm_8192` — and a context=1024-matched bytelm rerun
(`bytelm_xs_mtp4_ctx1024`) were queued but not completed before this
stop — rerun later if a clean baseline comparison against `qcutelm_vlt7`
is needed).

`qcutelm_vlt7` (the narrow-tokenizer/wide-codelm hybrid, see its own
module docstring for the full design) is being tested next, first config
`qcutelm_vlt7_bsq.py` — same bsq/dq=13/d_model=96+lm_d_model=256
architecture as `qcutelm_vlt6_grid_bsq_ntp_encdec` above, for a direct,
apples-to-apples comparison against its 2.5872 val_bpb.

## Session update — qcutelm_vlt7/vlt8: interleaved symmetric tokenizer,
## separate wide codelm, and a discovered window/code-value confound

**Lineage.** `qcutelm_vlt7` went through three design iterations within
the same file (see its module docstring for full detail):
- **v1**: codes interleaved directly in the byte stream (`t1 t2 c1 t3 t4
  c2 ...`), one shared stack playing both encoder and decoder roles, one
  loss. Motivated by making `qcutelm_vlt6`'s encoder/decoder input
  formats genuinely symmetric (v6's decoder uses a separate BOS-prepended
  format even when `shared_encoder_decoder=True`).
- **v2 (folded into v1's file)**: reserved a zero-vector slot in the
  encoder pass too, so both modes share sequence length and RoPE
  positions exactly — v1 had byte `t3` sitting at different positions in
  encoder vs. decoder mode, a real bug.
- **v3 (current `qcutelm_vlt7`)**: user caught that v1/v2 had *no
  advantage over bytelm at all* — a single shared stack run twice over
  ~the full byte-length sequence, at one width, is strictly worse than a
  same-width byte LM run once. Fix: reintroduce `qcutelm_vlt6`'s
  narrow-tokenizer/wide-`codelm` split — `codelm` is the only wide
  component and only ever processes the short `n_blocks`-length code
  sequence, giving `O(n_blocks²)` attention cost for the *entire* nominal
  context span instead of `O(context_len²)` — this is where qcute's
  compute argument actually lives, and v1/v2 had discarded it for
  architectural symmetry.

**`qcutelm_vlt7`'s current design**: Pass 1 (narrow tokenizer, "no-code"
mode — zero vector at every reserved slot) reads out each block's TRUE
code deterministically (no prediction, no `code_match`-style exposure
bias — see session discussion on this). `codelm` (separate wide weights,
short sequence only) forecasts the next code, trained via
`code_match_loss` (same mechanism as `qcutelm_vlt6`'s `CodeLM`, no new
loss invented). Pass 2 (same narrow tokenizer weights) decodes using
`codelm`'s *predicted* code, not the true one — the genuine generative
test. Ablations added: `attn_window` (ported `qcutelm_vlt6`'s verified
`O(T·window)` chunked attention, ~1.6-1.8x measured speedup),
`trainable_slot_embed` (learned vs. zero "no-code" marker),
`shared_tokenizer_phases` (untied Pass 1/Pass 2 weights, mirrors v6's
`shared_encoder_decoder=False`).

**Discovered confound → `qcutelm_vlt8`.** Across the entire
`qcutelm_vlt7_bsq` run, `code_conditioned_acc` and `no_code_acc` tracked
within ~0.1-0.4 percentage points of each other at *every* logged
checkpoint (e.g. step 3999/8000: 57.40% vs 57.33%) — no measurable
advantage from having the code at all. Root cause, diagnosed in-session:
`attn_window=64` is unrelated to the interleaved sequence's `(K+1)`
periodicity, so a chunk boundary can split a block's own bytes from its
own code slot (e.g. `context_len=1024, K=4`: block 12 spans positions
`[60,64]`, and a `window=64` chunk boundary falls exactly at position 64,
its code slot). More importantly, the chunked mechanism gives each
position ~`2×window` raw positions of reachable history — at
`window=64-80` that's ~16-32 *blocks* of direct raw-byte access, vastly
more than the single-block granularity a code is supposed to compress.
Since Pass 1 ("no-code") and Pass 2 ("forecast") share the same windowed
stack, Pass 1 can reconstruct nearly as much as Pass 2 just by reading
raw bytes still inside its window — the code becomes redundant with
information already available for free. **`qcutelm_vlt6` never had this
problem**: `decode_block` processes each block in total isolation (`[N,
K]`, batched, zero cross-block attention) — the code is the *only*
channel for anything beyond the current block, guaranteed by
construction. `qcutelm_vlt7`'s single-shared-stack unification is what
reintroduced the leak; `v6`'s asymmetric design never had a continuous
multi-block attention mechanism for it to leak through in the first
place. This is a second, independent mark against `v7`'s unification
(alongside the earlier-found "no compute saving without the narrow/wide
split") — not about efficiency this time, about whether the experiment
even measures what it claims to.

**`qcutelm_vlt8`** (forked from `v7`) fixes the alignment bug — `attn_window`
must now be a multiple of `K+1` (enforced at `Config` construction, not
just a config convention), so every chunk covers exactly whole blocks,
code slot included, no more accidental splitting. `qcutelm_vlt8_bsq.py`
uses `attn_window=80` (`16×(K+1)`, close to v7's 64) as a direct,
window-alignment-only comparison point against `v7`'s result.
`qcutelm_vlt8_bsq_tight_window.py` uses `attn_window=5` (`K+1`, `m=1`,
the tightest legal value) to directly test the confound hypothesis: does
`code_conditioned_acc` pull ahead of `no_code_acc` once the wide-window
raw-byte shortcut is closed off? (Residual gap even at `m=1`: the chunked
mechanism still gives one block of previous-chunk lookback, unlike
`v6`'s decode_block's zero cross-block attention — noted as a known
limit, not fully resolved by this ablation alone.)

**Queue** (sequential, one GPU job at a time, `caffeinate`-wrapped):
`qcutelm_vlt7_bsq` (done) → `qcutelm_vlt8_bsq` (running) →
`qcutelm_vlt8_bsq_tight_window` → `qcutelm_vlt7_bsq_trainable_slot` →
`qcutelm_vlt7_bsq_trainable_slot_untied` → `bytelm_xs_mtp4_ctx1024` →
`bpelm_8192` (8000 steps).

**Result — `qcutelm_vlt7_bsq`**: best val_bpb **2.4951** — beats every
`qcutelm_vlt6` grid cell (previous best 2.5872, `bsq_ntp_encdec`) and
lands within 0.008 of bytelm's own baseline (2.4872), the closest any
qcute variant has come to a real baseline this session. Despite this,
`code_conditioned_acc` vs `no_code_acc` stayed within ~0.3-0.6 percentage
points of each other for the entire run (e.g. final eval, step 7999:
51.92% vs 51.59%) — a small, only-recently-consistent gap in the code's
favor, not a strong signal — consistent with the window/code-value
confound diagnosed above (`attn_window=64` gives ~2×64=128 raw positions
of reachable history, ~25 blocks). Free-running generation quality stayed
poor throughout (1.56%-9.38% byte-match across gen checkpoints, no clear
upward trend), plausibly exposure bias compounding over the block-to-block
rollout — flagged, not yet diagnosed further. `qcutelm_vlt8_bsq` (same
architecture, block-aligned `attn_window=80`) and especially
`qcutelm_vlt8_bsq_tight_window` (`attn_window=5`, closes most of the
raw-byte shortcut) are the direct next data points on whether the good
bpb number reflects genuine code-based compression or is achieved mostly
through the wide-window raw-byte path.

Also noted in-session but not yet built: since `codelm`'s attention is
already dense `O(n_blocks²)` over the *entire* code history (cheap
precisely because `n_blocks = context_len/K`), predicting further than
one block ahead (`code[i+2]`, `code[i+3]`, ...) is a natural, low-cost
extension — an additional prediction head/shifted target on the same
already-computed representation, not a new architectural capability.
Flagged as a follow-up experiment, not implemented.

## Session update — supervision-edge taxonomy for qcutelm_vlt7/vlt8, and
## the actual goal: tokenizer/detokenizer-free codelm decoding

**Explicit goal stated mid-session**: the point of all the loss-composition
questions below is enabling `codelm` to run as a free autoregressive
generator in pure code space — init from some codes, roll forward feeding
its own predictions back as input (no re-encoding of bytes, no decoding
to bytes at every step), only detokenize once at the very end. This
reframes every open design question below: does a given loss help close
the gap between `codelm`'s own predictions and the true encode-phase code
distribution (`z_hat_enc`), since that gap is exactly what determines
whether feeding predictions back as inputs stays stable over a long
free-running rollout?

**Current loss composition, precisely** (confirmed against
`qcutelm_vlt8.py`'s `forward()`, `code_match_weight=1.0` active in
`qcutelm_vlt8_bsq.py`):
```
loss = loss_nocode + loss_decode + code_match_weight * code_match_loss
```
Three losses active, not one — `codelm`'s own prediction *does* get a
direct target (`code_match_loss`). What has **no direct target at all**
is the *encoder's* own code (`code_pre`'s output, `z_hat_enc`) — it only
ever receives gradient indirectly, through the long chain `z_hat → codelm
input → forecast → z_proj → decode → NTP loss → backprop`. This is the
real gap, not "codelm doesn't care" (it does) but "nothing directly
shapes what the encoder's code itself should look like."

**Supervision-edge grid** (source of target × who gets trained):

| Source ↓ \\ Target → | Encode (`code_pre`) | Decode (bytes) | CodeLM |
|---|---|---|---|
| Ground truth bytes | *(indirect only)* | ✅ existing, always-on (both phases) | *(indirect only)* |
| Encode's own code (`z_hat_enc`, detached) | — (self) | 🆕 missing — this is `qcutelm_vlt6`'s `aux_recon_weight` (same-block decode, short direct gradient path for `code_pre` — motivated by the earlier-diagnosed BSQ `code_proj` gradient weakness, 10-28x, from `F.normalize`'s Jacobian) | ✅ existing — `code_match_loss` |
| Decode phase's code (`z_hat_dec`, doesn't exist yet — would need a new readout on Pass 2's hidden states) | 🆕 proposed | n/a | 🆕 rejected — see below |
| CodeLM's own prediction (`pred_soft`, detached) | 🆕 proposed (`encode_match_weight`) | ✅ existing — Pass 2 "forecast" mode | — (self) |

**`codelm`'s target: encode-phase, not decode-phase — decided, not just preferred.**
Given the free-decoding goal, pointing `code_match_loss` at a hypothetical
`z_hat_dec` (computed under decode-phase conditioning, itself dependent on
`codelm`'s own past output) would be actively counterproductive: (1)
`codelm`'s *input* mechanism (`in_proj`/`z_proj`) is always trained on
`z_hat_enc`-distributed values, so a `z_hat_dec` target would train
`codelm`'s output into a different distribution than its own input expects
— free-rolling would get *worse*, not better; (2) `z_hat_dec` would be a
moving target (shifts as `codelm` itself trains, since it depends on
`codelm`'s conditioning), reintroducing the exact moving-target
instability `qcutelm_vlt4`'s `--joint_lm` needed a `code_lm_weight`
warmup schedule to work around. Keeping the target as `z_hat_enc` (already
the case) is correct and not up for revision.

**Two build candidates, reprioritized against the free-decoding goal**:
- **`aux_recon_weight`** (Encode←own code, short path): sharpens
  `z_hat_enc` itself — the canonical space `codelm` is trying to imitate.
  Helps.
- **`encode_match_weight`** (Encode←CodeLM's prediction, mutual
  consistency): directly trains the gap free-rolling depends on being
  small (`codelm`'s prediction ↔ true `z_hat_enc`). Arguably the single
  most relevant unbuilt piece given the stated goal, not just a nice-to-have.
- (A single-pass redesign was also discussed — collapsing Pass 1/Pass 2
  into one causal pass, recovering real compute savings the way `v7`/`v8`'s
  two-pass scheme doesn't. Deprioritized: it's about efficiency and
  architectural cleanliness, orthogonal to the free-decoding goal, and
  makes maintaining a clean condition-free `z_hat_enc` harder since encode
  and decode would share causal hidden states instead of running as an
  isolated pass.)

**Both built.** `aux_recon_weight` added a new `decode_block_local` method
(ported from `qcutelm_vlt6`'s `decode_block` — block-local, `[N,K]`
batched, zero cross-block attention, reuses `dec_byte_emb`/`run_dec_blocks`/
`dec_head`/`z_proj`; the chunked-window check in `CausalSelfAttention`
auto-falls-back to dense SDPA for this short a sequence, so no special-
casing needed there). `code_pre`'s output is no longer collapsed straight
into `quantize()` — split into `pre_q` (pre-quantization) so
`encode_match_weight` can target it directly:
`encode_match_loss = F.mse_loss(pre_q[:, 1:, :], pred_soft.detach())`.
Both smoke-tested (all 5 weight combinations: baseline, `aux_recon` only,
`encode_match` only, both, `code_match_weight=0`) and overfit-verified
(`code_acc`→100%, `aux_recon_acc`→~75%, both loss terms decreasing) before
touching the live queue.

**Queue reprioritized** (per explicit instruction — let `qcutelm_vlt8_bsq`
finish, then an NTP-only baseline, then dense supervision, ahead of the
previously-queued ablations): `qcutelm_vlt8_bsq` (running) →
**`qcutelm_vlt8_bsq_ntp_only`** (`code_match_weight=0` — codelm gets *zero*
direct supervision, only indirect gradient via Pass 2's backprop chain,
i.e. "how a regular LM trains its last layer" with no auxiliary
intermediate targets) → **`qcutelm_vlt8_bsq_dense_supervision`** (all four
edges active: `code_match_weight=aux_recon_weight=encode_match_weight=1.0`)
→ `qcutelm_vlt8_bsq_tight_window` → the two `qcutelm_vlt7` ablations →
baseline reruns. Three-way comparison once these land: does removing
codelm's direct target (`ntp_only`) hurt val_bpb noticeably, and does
adding the two new edges (`dense_supervision`) help — both against
`qcutelm_vlt8_bsq`'s own result as the middle reference point.

## Session update — symmetry flaw found in qcutelm_vlt7/vlt8; qcutelm_vlt8
## default flipped to untied; qcutelm_vlt9 built (true symmetry, slow prefill)

**The flaw** ("i found a flaw in v8 for symmetry, better untie tokenizer
detokenizer"): `qcutelm_vlt7`/`vlt8` were never actually symmetric between
encode (Pass 1) and decode (Pass 2) despite the "symmetric" framing that
motivated sharing their weights. Decode is a genuine CONSUMER of
`codelm`'s output — every block's byte generation is conditioned on a
predicted code. Encode is a pure PRODUCER — it computes `z_hat` from raw
bytes alone and never consumes `codelm`'s output in return. That's a
one-way pipeline (`encode -> codelm -> decode`), not a symmetric pair.
True symmetry would require encode to ALSO consume `codelm`'s forecast as
conditioning — which needs a genuine block-by-block AR handshake between
encoder and `codelm` (encode's computation for block `i` needs `codelm`'s
forecast built from block `i-1`'s TRUE code, which needs encode to have
already finished block `i-1`) — expensive (loses full-sequence parallel
training) and circular for no benefit `qcutelm_vlt7`/`vlt8` were actually
using. Given encode and decode are different functions (unconditional vs.
conditional LM), sharing weights between them was asking one set of
weights to serve two incompatible jobs.

**`qcutelm_vlt8`'s default flipped**: `shared_tokenizer_phases` now
defaults to `False` (untied) — promoted from "one ablation among several"
to the theoretically-motivated default. `qcutelm_vlt8_bsq_untied.py`
(same architecture as `qcutelm_vlt8_bsq.py`, only this flag differs)
queued as the direct comparison point. The three already-designed
ablations (`ntp_only`, `dense_supervision`, `tight_window`) were pinned to
`shared_tokenizer_phases=True` explicitly, preserving their original
single-variable-isolation intent against `qcutelm_vlt8_bsq`'s own
(pre-flip) result — the default change happened after they were written
and queued.

**`qcutelm_vlt9` built**: genuine architectural symmetry — encode and
decode are now literally the SAME function (`SymmetricLM`, one set of
weights), every block structured as `[code_prefix, K bytes]` where
`code_prefix` is a bootstrap marker (block 0, nothing precedes it) or
`codelm`'s forecast (every later block, built from the TRUE codes of all
earlier blocks). Resolved via a genuine `n_blocks`-iteration Python loop
— no way to vectorize away, since block `i`'s input literally cannot be
constructed until block `i-1`'s true code is known. `codelm`'s own
supervision (`code_match_loss`/`aux_recon_weight`/`encode_match_weight`,
all ported unchanged) is computed via one final vectorized
`codelm(z_hat_full)` call after the loop — `codelm`'s own attention is
causal, so its per-position predictions from that one call are identical
to what the in-loop incremental calls gave, avoiding the need to cache
per-step `codelm` state.

**Problem, flagged before building, not discovered after** ("problem v9
is prefill slow"): measured directly, `n_blocks=32` already takes
**~2.1s/train-step on CPU** (batch=4); the loop is roughly `O(n_blocks²)`
(each iteration recomputes attention over the growing sequence), so
`n_blocks=256` (matching `qcutelm_vlt7`/`vlt8`'s usual `context_len=1024`)
would be ~64x slower — days for an 8000-step run, not hours. Queued
config (`qcutelm_vlt9_bsq_small.py`) deliberately trades scale for
tractability: `context_len=128` (`n_blocks=32`), `batch_size=8`,
`steps=2000` — same `K`/`dq`/`quant_type`/`d_model`/`lm_d_model` as the
other `bsq` runs (architecture held constant), not meant to be
bpb-competitive with the full-scale runs, just enough to see whether true
symmetry changes `code_conditioned_acc`/`within_block_acc` trends at all.
One structural tradeoff worth noting: `qcutelm_vlt9` loses `qcutelm_vlt7`/
`vlt8`'s free no-code-vs-code baseline ablation (no more separate
unconditioned Pass 1) — adding one back would reintroduce a second full
pass, undoing the point of this fork.

All three new/changed pieces (`qcutelm_vlt8`'s default flip,
`qcutelm_vlt8_bsq_untied.py`, `qcutelm_vlt9.py`) smoke-tested (forward/
backward, all params get gradient, `generate()` runs) and overfit-verified
(`qcutelm_vlt9`: `code_conditioned_acc`→90.6% on a tiny fixed batch,
genuinely learning) before touching the live queue. Full updated queue:
`qcutelm_vlt8_bsq` (running) → `qcutelm_vlt8_bsq_untied` →
`qcutelm_vlt8_bsq_ntp_only` → `qcutelm_vlt8_bsq_dense_supervision` →
`qcutelm_vlt8_bsq_tight_window` → the two `qcutelm_vlt7` ablations →
`bytelm_xs_mtp4_ctx1024` → `bpelm_8192` → `qcutelm_vlt9_bsq_small`.

**`qcutelm_vlt8_bsq` result**: best val_bpb **2.4771** — an improvement
over `qcutelm_vlt7_bsq`'s 2.4951 (the block-aligned windowing fix helped,
as hypothesized), now the closest any qcute variant has come to bytelm's
2.4872 — actually *beating* it, the first qcute config to do so this
session.

**Two operational bugs caught and fixed immediately after `v8_bsq`
finished:**
1. **Queue-editing assumption was wrong.** This session's earlier
   reasoning ("flat sequential bash scripts are read line-by-line, safe
   to edit not-yet-reached lines without restarting the orchestrator") was
   incorrect — when `qcutelm_vlt8_bsq` finished, the queue jumped straight
   to `qcutelm_vlt8_bsq_tight_window`, silently skipping `untied`,
   `ntp_only`, and `dense_supervision` (all inserted via in-place edits
   while the script was running). Caught within seconds via the live
   monitor output. Fix: rebuilt the queue script from scratch with all 9
   remaining experiments verified present (`grep -c "STARTING"` = 9)
   before relaunching — going forward, always fully restart the
   orchestrator after any edit, no exceptions, regardless of script
   structure.
2. **`init_head_bias_to_unigram` didn't initialize `dec_head`'s bias when
   untied.** With `shared_tokenizer_phases=False`, `dec_head` is a
   separate `nn.Linear` from `head` — the unigram-frequency bias init only
   ever touched `model.head.bias`, leaving `dec_head.bias` at its random
   default. Caught by comparing `qcutelm_vlt8_bsq_untied`'s very early
   `code_acc` (0.02-0.15% at step ~20) against `qcutelm_vlt8_bsq`'s
   equivalent point (~12-14%) — too large a gap to be normal variance.
   Fixed in both `qcutelm_vlt7.py` and `qcutelm_vlt8.py` (same bug in
   both, `qcutelm_vlt7_bsq_trainable_slot_untied` was still queued and
   would have hit it too): `init_head_bias_to_unigram` now also copies the
   bias into `dec_head` when it's a distinct module. Verified via a direct
   before/after equality check, then the untied run was killed (~30 steps
   in, negligible loss) and relaunched with the fix — `code_acc` recovered
   to ~12-14% by step ~30, in line with expectations.

**`qcutelm_vlt8_bsq_untied` result: best val_bpb 2.4462** — better than
`qcutelm_vlt8_bsq`'s 2.4771 (shared weights) and clearly ahead of
bytelm's 2.4872. This is the first *quantitative* confirmation that the
symmetry-flaw finding was correct, not just architecturally cleaner —
untying encode/decode weights measurably improved val_bpb. Interesting
transient during training: `no_code_acc` led `code_conditioned_acc` by as
much as ~4.5pp in the first ~500 steps (decode's `dec_head`/`dec_blocks`
learning from scratch without the head-start shared weights would have
given), before `code_conditioned_acc` overtook it by ~step 1200 and
stayed ahead for the rest of training — consistent with the untied
decode path needing a few hundred steps to catch up, then benefiting from
not fighting encode for representational capacity. Two full runs now
confirm the same pattern: `qcutelm_vlt7_bsq` (2.4951) → `qcutelm_vlt8_bsq`
(2.4771, window-alignment fix) → `qcutelm_vlt8_bsq_untied` (2.4462,
symmetry fix) — each independent fix improved val_bpb, in the order they
were found.

**`qcutelm_vlt8_bsq_ntp_only` result: best val_bpb 2.4885** —
`code_match_weight=0`, codelm gets zero direct supervision. Worse than
both `qcutelm_vlt8_bsq` (2.4771) and `qcutelm_vlt8_bsq_untied` (2.4462),
landing essentially at bytelm's own baseline (2.4872) — i.e. removing
`code_match_loss` erases qcute's entire margin over the plain byte LM
baseline. Confirms `code_match_loss` provides real, measurable value:
codelm does not learn a useful forecast purely through the long indirect
backprop chain (Pass 2's byte NTP loss → `codelm` → `z_hat`) — it needs
the direct target.

**Live finding during `qcutelm_vlt8_bsq_dense_supervision`: likely
mutual-collapse between `code_match_loss` and `encode_match_weight`.**
`code_match_loss` dropped to exactly 0.0000 within ~400 steps;
`encode_match_loss` has stayed very small (~0.005-0.01) throughout —
together suggesting `codelm`'s predictions and the encoder's own code
(`z_hat`) have converged *toward each other*, exactly the risk flagged
when `encode_match_weight` was first proposed ("both could converge
toward a trivial, easily-mutually-predictable constant code that carries
no real byte information"). With both directions active simultaneously
at weight 1.0, nothing anchors the code to stay diverse across blocks.

Diagnostic signal: `aux_recon_acc` has stalled flat around 28-30% for 7+
consecutive evals while every other metric climbs to ~44-45%.
`aux_recon` is the one metric that would visibly expose this — it
reconstructs a block's bytes using *only* that block's own true code,
zero cross-block attention, genuinely isolated (unlike Pass 1/Pass 2's
windowed multi-block stack). If codes have collapsed toward similar
values across blocks, `aux_recon` can't distinguish blocks from their
codes alone. `no_code_acc`/`within_block_acc` are unaffected (never
depend on code diversity — no-code mode always uses zero, relying on the
raw-byte window instead). `code_conditioned_acc` climbing in lockstep
with `no_code_acc` (not pulling ahead) suggests decode isn't actually
leveraging the (now-degraded) code either, likely masked by the same
wide-window raw-byte shortcut diagnosed earlier this session
(`attn_window=80` gives ~32 blocks of direct reachable history) — an
escape hatch `aux_recon` doesn't have. Hypothesis: the window confound
and this mutual-collapse risk may be compounding — the wide window masks
a code-collapse problem that `qcutelm_vlt8_bsq_tight_window` (already
queued next) would likely expose in `code_conditioned_acc` too, the same
way `aux_recon_acc` is exposing it here. To be confirmed once both runs
finish.

## `qcutelm_vlt10` — Clockwork-RNN-inspired multi-timescale sandwich (new fork, built)

Motivated directly by the mutual-collapse finding above: `qcutelm_vlt8`'s
`aux_recon_weight`/`encode_match_weight` reconcile `codelm` and the
tokenizer's code only via auxiliary losses with a long backprop path —
observed to be capable of collapsing to a trivial mutually-predictable
solution instead of a genuinely informative code. `qcutelm_vlt9` (block-
by-block true symmetry) is a different, still-live answer to a related
but distinct problem (encode/decode role asymmetry) and is not touched by
this fork.

`qcutelm_vlt10`'s bet: make `codelm` a literal middle LAYER of one tiered
stack — LOWER tokenizer layer (every byte, "fast clock") produces codes
via strided readout every K bytes -> CODELM (sparse "slow clock", every K
bytes) forecasts the next code -> UPPER tokenizer layer (every byte,
"fast clock", re-synced from CODELM's forecast at every block-start byte
position, substituted in place of the lower layer's own hidden state
there — block 0 excepted, no forecast exists yet). Only the upper layer
has a loss (byte NTP over the whole sequence) — the lower layer has no
loss/head of its own (session-confirmed default: "at timesteps not
modulo k, skip to upper layer", so a separate lower-layer loss would
train a representation nothing downstream is forced to use consistently
at most positions). `code_match_loss` unchanged, still the sole training
signal for `codelm` itself.

Crucially, unlike `qcutelm_vlt9`, this needs NO sequential python loop:
the lower layer's pass depends only on raw bytes (causal, one vectorized
call), `codelm`'s forecast at block i depends only on codes <i (already
available from that one lower-layer pass), and the upper layer's input
(lower's hidden states with the block-start substitution) can be built
functionally (cat/where, no in-place mutation, autograd-safe) and run in
one more vectorized call. Three full-sequence passes total, no
`O(n_blocks^2)` loop — the design's actual payoff over `v9`.

Windowing: both tokenizer tiers AND `codelm` now get windowed attention
(`attn_window`/`lm_attn_window`, both default 64 per the design spec) —
new for `codelm`, which was always dense in `v7`/`v8`/`v9` since its
sequence was already short; here `codelm`'s window is still the largest
in *effective* raw-byte coverage since each of its tokens already
represents K bytes.

Sanity-checked (tiny CPU forward+backward, dense and windowed variants,
context_len=32): loss/metrics compute correctly, zero parameters with
missing gradients, `generate()` runs for both configs (recomputes the
full lower+codelm+upper pipeline from scratch every generated byte, no
KV cache, consistent with this lineage's existing simplicity tradeoff;
works for any prefix length, not just multiples of K).

`configs/qcutelm_vlt10_bsq.py` created (same K/dq/quant_type/d_model/
lm_d_model as the `v7`/`v8` bsq runs for direct comparability,
`attn_window=lm_attn_window=64`) — not yet queued (only one training job
at a time; current `run_vlt7_queue.sh` still has `dense_supervision`,
`tight_window`, both `vlt7` trainable-slot ablations, both baselines, and
`qcutelm_vlt9_bsq_small` ahead of it).

Noted as a documented future extension, not built: generalizes to N
levels by chaining — `codelm_1` (period K1) between lower and a mid
layer, `codelm_2` (period K1*K2, operating on `codelm_1`'s own code
stream rather than raw bytes) between mid and upper, and so on, each
level's `code_match_loss`-style regularization targeting the level below
it (a hierarchy of self-consistency losses).
