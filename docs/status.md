# Status

Tracks progress against the phase plan in
[continuous_tokenizer_handover.md §5](continuous_tokenizer_handover.md#5-implementation-phases).

## Phase 0 — infrastructure

- [x] Byte-level data pipeline (`datasets/enwik8.gz` + `datasets/enwik8_tiny.gz`
      500,000-byte subset via `scripts/prepare_data.py`).
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
  | `bytelm` (mtp_heads=4) | `bytelm_xs_mtp4_converged` | **2.52** | 1300 | 3.4M |
  | `bpelm` (vocab=8192) | `bpelm_8192_converged` | **2.35** | 300 | 5.25M |
  | `qcutelm` (BSQ, K=4) | not yet re-run at K=4 (below is the older K=8 run) | 4.97 | 4750 | 3.9M |

  Plots: `logs/bytelm_xs_mtp4_converged/bpb.png`,
  `logs/bpelm_8192_converged/bpb.png` (`scripts/plot_run.py`). Both bytelm
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
      `logs/qcutelm_bsq_k4_converged/bpb.png`.
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
    in a poor local optimum instead. `logs/qcutelm_bsq_k4_tightlycoupled/bpb.png`.
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
    `bytelm_xs_mtp4_converged`, `bpelm_8192_converged`, and
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
    | `qcutelm_bsq_k4_converged` | loosely-coupled (historical) | 5.54 | 5400 | 72.76% | 60.66% |
    | `qcutelm_bsq_k4_tightlycoupled` | tightly-coupled, no aux | 7.12* | 400 | 13.48% | 90.34% |
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
    (pre-MaskGIT) run's 5.39. `logs/qcutelm_maskgit_lfq_aux/bpb.png`. Next:
    combine MaskGIT with uncertainty weighting alone (no entropy_reg, given
    that combination's demonstrated instability) to see if the two fixes
    stack — not yet run.
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
      `--entropy_reg_weight`, `--bsq_sample_generation`, `--maskgit_T`,
      `--pretrain_ae`/`--init_encoder_decoder_from`.
- [ ] Not yet done: extracting the model-agnostic pieces (`Logger`,
      `Checkpointer`, `load_config_module`, `load_enwik8`, `split_train_val`,
      `lr_at`, RoPE math) into a shared `qcute/utils.py` — identified as safe
      to share (see module docstrings) but deferred as a separate decision.
