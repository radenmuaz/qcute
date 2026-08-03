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
      same tiny 500KB subset, same warmup+constant LR schedule:

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
