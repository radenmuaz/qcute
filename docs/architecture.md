# Architecture

Three self-contained modules in the `qcute/` package: `qcute/bytelm.py`,
`qcute/qcutelm.py`, and `qcute/bpelm.py`. None import each other or share a
common name with the package itself (an earlier top-level `qcute.py` next
to the `qcute/` package collided on `import qcute` — renamed; each module
inlines its own tiny data-loading/Logger/Checkpointer helpers rather than
importing from the others, to keep them independent). Merge shared pieces
into a `qcute/utils.py` once there's a fourth reason to (see each module's
docstring for which pieces are safe-to-share candidates vs. deliberately not).

See [continuous_tokenizer_handover.md](continuous_tokenizer_handover.md) for
the full design this implements, and [status.md](status.md) for what's done.

Both scripts' `main()` follow the same (duplicated, not shared — see
docstrings for the extraction candidates identified but not yet acted on)
conventions:

- **Run naming**: every run gets a `run_name`, resolved as `--run_name`
  (explicit) → `--config`'s filename stem → a script-specific default
  formula (`bytelm_<preset>_<timestamp>`, `qcutelm_<bottleneck>_<timestamp>`,
  `bpelm_<timestamp>`). Logging and checkpointing both key off the same
  `run_name`, so everything for one run lives under one findable name.
- **Logging**: a `tqdm` progress bar for live terminal feedback; at
  `--log_every`/`--eval_every`, `Logger` writes the same line to *both*
  `logs/<run_name>/run.log` (raw text, `tail -f` from another terminal) and
  `logs/<run_name>/run.jsonl` (structured, one record per line, prefixed
  with elapsed `[HH:MM:SS]` / `elapsed_s` rather than a raw epoch
  timestamp). `logs/` is gitignored.
- **Config files**: `--config path/to/file.py` loads a plain Python module
  (see `configs/`) whose module-level variables become new argparse
  defaults (`load_config_module` + `parser.set_defaults(...)`, filtered to
  known flags) — **CLI flags still override config file values**, which
  override the script's hardcoded defaults. Two-pass parse: a small
  pre-parser reads just `--config` before the full parser is built.
- **Checkpointing**: `Checkpointer` saves `checkpoints/<run_name>/best.pt`
  (overwritten only when the tracked val metric improves) and
  `checkpoints/<run_name>/last.pt` (overwritten every `--save_every_n_evals`
  eval calls, default every eval). Each checkpoint carries model + optimizer
  state, step, and `cfg` as a dict (`dataclasses.asdict`) so `--eval_only
  --checkpoint_path ...` can rebuild the exact architecture without needing
  `--preset`/`--bottleneck` etc. repeated. `checkpoints/` is gitignored.
- **`--eval_only`**: skips training entirely, loads `--checkpoint_path`,
  runs one eval pass, then falls through to the same qualitative-generation
  and benchmark blocks a normal run would hit at the end — implemented as
  an `if args.eval_only: ... else: <training loop>` split, not an early
  `return`, so those tail blocks aren't duplicated.
- **Qualitative generation** (`--qual_gen_bytes > 0`): draws a prompt from
  `--qual_source` (`train`/`val`/`user` — `--qual_user_text` for the last),
  generates a continuation, and — when the prompt was dataset-drawn — logs
  the real ground-truth continuation alongside it plus the model's bpb on
  that truth (`score_continuation_bpb`, teacher-forced, isolated to the
  continuation region only). This is a qualitative complement to the
  aggregate val_bpb number, not a replacement for it.

## `qcute/bytelm.py` — byte-level baseline LM (with MTP head)

The Phase 0 "number to beat" (handover §5, Phase 0) — but specifically the
*strong* one, BPE+MTP (handover §1.6), adapted to raw bytes: a causal
transformer with `mtp_heads` parallel output heads predicting bytes
t+1..t+n from one trunk pass, bandwidth-matched to `qcute.qcutelm`'s `K`
and `qcute.bpelm`'s bytes/token so BPB and generation latency are
comparable at matched bandwidth across all three. `mtp_heads`/`K` default
to **4** at tiny-corpus scale (the `xs` preset), not the handover doc's
8 — empirically, BPE (the fair comparison point) only reaches ~3-4
bytes/token on a 900,000-byte corpus before larger vocabs start
memorizing phrases (see `scripts/train_bpe.py`'s docstring), so targeting
8 there would be a bandwidth mismatch specific to this corpus scale.
`sd`/`md` keep 8, matching the doc's full-corpus-scale default.

- `ByteLM`: pre-norm transformer blocks, RoPE on Q/K (`rope_cos_sin` /
  `apply_rope`), `F.scaled_dot_product_attention(is_causal=True)`,
  GELU MLP (4x hidden). `heads` is an `n`-length `ModuleList`; only head 0
  (immediate next-byte) is weight-tied to the input embedding, matching
  standard MTP practice — the other heads are untied.
- Init: GPT-2-style — `N(0, 0.02)` for all `Linear`/`Embedding` weights, with
  the residual-stream projections (`attn.out`, `mlp.down`) additionally scaled
  by `1/sqrt(2 * n_layers)`. **Load-bearing**: PyTorch's default `Embedding`
  init (`std=1`) blows up logits at init to ~1000 bits/byte instead of the
  correct ~8 (`log2(256)`); don't remove this without re-verifying init-time
  bpb is sane.
- `mtp_loss()`: mean CE across all `mtp_heads` (the training signal) plus
  head-0 BPB alone (the metric comparable to a plain non-MTP byte LM and to
  `qcute.qcutelm`'s BPB) — exact, no ELBO needed, unlike the tokenizer's
  quantized bottleneck.
- `batch_iter()` fetches `context + mtp_heads` bytes per window (instead of
  `context + 1`) so every head has a real target; `split_train_val()` /
  `eval_bpb()` add a held-out val split with periodic evaluation.
- Presets in `PRESETS` (power-of-2-friendly dims): `xs` ≈3.7M (d=256,
  4 layers, 4 heads, ctx 256 — sized to roughly match `qcute.qcutelm`'s
  param count, for quick local runs), `sd` ≈101M (d=1024, 8 layers,
  16 heads, ctx 2048), `md` ≈403M (d=2048, 8 layers, 16 heads, ctx 2048).
  Param count ≈ `12 * d_model^2 * n_layers` (vocab is only 256, so embedding
  params are negligible). `--context`/`--mtp_heads` override a preset's
  values for experimentation.
- `generate_ar()` / `generate_speculative()` / `benchmark_generation()`:
  plain one-byte-per-step AR decoding vs. **self-speculative decoding**
  (Leviathan et al.-style) using the model's own MTP heads as the draft
  proposer, verified by one true causal forward pass, accept/reject via
  standard speculative rejection sampling. Lets one verification step emit
  up to `mtp_heads` bytes — this is the fair latency comparison against
  `qcute.qcutelm`, which emits `K` bytes per LM+decoder step by
  construction. Batch size 1 only. `--benchmark_generate_bytes` runs both
  and reports bytes/sec, average accept length, and speedup.
- `score_continuation_bpb()`: one causal forward over prompt+continuation,
  head-0 CE restricted to the continuation positions — exact, no chunking
  concerns (this module is byte-level throughout).

## `qcute/qcutelm.py` — end-to-end tokenizer + latent LM

Encoder + bottleneck + LM + decoder, trained jointly, with a generation loop —
collapses Phase 1 (bottleneck) and Phase 2 (LM interface) into one script.
Simplified vs. the handover doc on purpose: encoder/decoder are plain MLPs
over one fixed K-byte chunk (non-streaming, no causal-SSM byte context —
the doc calls this the "naive" chunk-local design with boundary artifacts,
§1.3/§1.4, but it's the fastest path to something end-to-end and trainable).
Supersedes the old streaming-causal-encoder Phase 1 autoencoder, archived at
`archive/tokenizer_phase1_standalone_autoencoder.py`.

- `FSQ` / `BSQ` (handover §1.2.2 / §1.2.3): selectable via `--bottleneck`.
  FSQ default `dq=6, L=8` (codebook `8^6`); BSQ default `dq=18` (codebook
  `2^18`). Both straight-through. `bsq_quantize(v, dq, lfq)` is BSQ's
  quantization math (normalize + sign STE) factored out to a module-level
  function — reused by both the encoder's `BSQ.forward` (applied to its own
  learned projection) and the LM's output head (applied directly to the
  LM's raw predicted latent, no separate learned projection) — same
  quantization boundary, two different producers. `lfq=True` (`--lfq`)
  regresses BSQ to plain LFQ (Yu et al. 2023): skip the L2-normalize-onto-
  hypersphere step, sign the raw projection directly (hypercube corners
  `{-1,+1}^dq`, unscaled, vs. BSQ's hypersphere corners `||z_hat||=1`).
  Sign bits (targets) are identical either way, only `z_hat`'s scale
  changes — but that scale difference turned out to matter a lot in
  practice (see status.md's gradient-norm diagnosis: LFQ's larger `z_hat`
  magnitude produces vastly larger decoder gradients than BSQ's).
  `bsq_entropy_reg(v)` is the LFQ/BSQ-paper entropy regularizer (Yu et al.
  2023 §3.2) neither paper's technique actually works without — minimize
  per-example bit entropy, maximize batch-averaged bit-usage entropy,
  countering the code-usage collapse that (per status.md's qualitative-
  generation finding) this project hit in practice. Computed on the LM's
  raw predicted latent (`v_pred`), not the encoder's; wired in as a
  training-loop-only term (see below), not baked into the model.
- `ChunkEncoder` / `ChunkDecoder`: per-position `byte_emb + pos_emb`, then
  stacked **`MixerBlock`**s (mixer + optional post-mixer MLP, pre-norm
  residual) before a final projection. Depth defaults to the shared
  `cfg.tokenizer_layers` (`--tokenizer_layers`, default 1) for both, with
  independent overrides `--encoder_layers`/`--decoder_layers` (each `None`
  by default, falling back to the shared value) for asymmetric designs —
  `encoder_layers=0` is valid, no `MixerBlock`s at all, pure flatten →
  `Linear` → bottleneck. A tiny-subset overfit sanity check (see status.md)
  found that 1 layer caps train `recon_acc` at ~88-97% no matter how
  width/LR/weight-decay are tuned, while 2 layers reaches a clean 100% —
  a real capacity ceiling at that scale, not a training-dynamics issue;
  at full-corpus scale, though, an extensive depth/width/asymmetry sweep
  (status.md) only reached 83% best-case, still short of 95%.
  `--mixer {attention,conv}` selects the cross-K-position mixer (also with
  independent `--encoder_mixer`/`--decoder_mixer` overrides):
  `FullSelfAttention` (full non-causal self-attention over the K
  positions) or `FullConvMixer` (a single non-causal 1D conv,
  `kernel_size=2K-1` with symmetric `padding=K-1` — *not* `kernel_size=K`,
  which was tried first and is wrong: it gives only one output position
  full coverage of all K inputs, the rest get an inconsistent partial
  window; `2K-1` with symmetric padding guarantees *every* output
  position's receptive field spans the entire real input range, matching
  attention's actual coverage — see `FullConvMixer`'s docstring).
  `--disable_mixer_mlp` tests the mixer alone. Deliberately non-causal — a
  causal-TCN encoder variant was tried and reverted earlier (see
  status.md) since the LM, not the chunk-local encoder/decoder, is what
  owns causality here; the chunk is always fully observed, so hiding
  future bytes from earlier positions bought nothing.
  `--context_len` (default 0) + `gather_left_context()`: optional
  asymmetric left-side byte context for the encoder only — up to K extra
  bytes from the *previous* chunk's tail, still causal.
  **`ChunkDecoder` is now code-only** (`forward(z)`, no second argument):
  one forward pass reconstructs all K positions from `z_proj(z)` broadcast
  identically to every position, no masked-byte input, no iterative
  refinement (`decode_bytes(decoder, z)` is a trivial one-shot
  `decoder(z).argmax(-1)`, replacing the old `maskgit_decode`).
  **History**: an earlier version was MaskGIT-style (handover §1.4b) — a
  masked-byte-level input (`byte_emb` sized `vocab+1`, `MASK` token), with
  training sampling a cosine mask-rate schedule (`maskgit_mask`) and
  inference using T-step confidence-based refinement. A real bug was found
  and fixed in *that* version first: its `forward()` had applied a plain
  per-position `Linear`/`GELU`/`Linear` MLP to `[N, K, d_byte]`, which
  `nn.Linear` broadcasts over all but the last dimension — **zero**
  cross-position mixing despite the docstring claiming otherwise (the
  entire point of MaskGIT). Fixing that (adding real `MixerBlock` mixing)
  roughly doubled `recon_acc` (13.5%→24-26%, `qcutelm_joint_bsq_noaux_fixedmix`)
  versus the identical pre-fix config. But even after that fix, a later
  ablation (training with `full_mask` — always 100% masked, the "hardest
  case") found the decoder's *only* real input under that condition was
  `z_proj(z)` — the masked-byte channel had degenerated to contributing
  nothing chunk-specific — meaning the two-input design had already
  effectively become "codes only", just with dead weight (`byte_emb`,
  `mask_id`) still attached. Removing it entirely also matches MaskGIT's
  own two-stage assumption (Chang et al. 2022): the tokenizer trains to
  convergence and is *frozen* before any masked-token scheme is
  introduced — MaskGIT never masks inside tokenizer training, which is
  what this file had drifted into. If masking is wanted at all now, it's
  `--pretrain_encoder_mask` on the *encoder's* raw byte input
  (`mask_bytes()`, denoising-autoencoder-style — decoder still grades
  against true uncorrupted bytes), not the decoder.
  `Config.quant_grad_scale` (default 1.0, no-op) + `GradScale`: QAT-style
  backward-only gradient rescale at the encoder's BSQ quantization
  boundary — implemented to counteract encoder/decoder gradient-norm
  imbalance found by a side diagnostic (status.md), not yet tested at a
  nonzero value in a real training run.
  `init_decoder_bias_to_unigram()`: initializes `head`'s bias to the
  training corpus's log unigram byte frequency (GPT-2-style embedding
  init) — called unconditionally at the start of `pretrain_autoencoder()`,
  a free head start (initial loss starts near the unigram floor instead
  of the uniform floor).
- `LatentLM`: causal transformer over the code sequence, RoPE on Q/K
  (duplicated from `qcute/bytelm.py` rather than imported — see below).
  Three input/output modes:
  - **Continuous** (default): input is a linear projection of `z_hat`;
    output is FSQ's per-dim categorical logits or BSQ's raw `dq`-dim
    latent (no discrete embedding table — BSQ's implicit codebook, `2^dq`
    = 262144 at the default `dq=18`, would need a ~67M-param table).
  - **Discrete vocabulary** (`vocab_size=<int>`, used by `train_vocab_lm`
    below): input is a plain `nn.Embedding(vocab_size, d_model)`, output is
    a weight-tied categorical softmax over the vocab — exactly like
    `qcute.bpelm`/`bytelm`. Only viable once a tokenizer is frozen (see
    `build_code_vocab` below); has a real OOV/UNK problem (see status.md).
    `main()`'s `--freeze_after_pretrain` branch selects between this and
    the factorized/PQ path below via `--pq_groups` (int, default 1): `1`
    (special case) routes to this vocab-table path; any other value routes
    to `train_factorized_lm`/`encode_to_codes` (below) instead. Independent
    of `--lm_factorized_input`, which governs the *unfrozen* joint-training
    LM's input format, a separate setting.
  - **Factorized/PQ** (`--lm_factorized_input`, `FactorizedCodeEmbedding`):
    input is a compositional embedding — one independently-learned vector
    per `(dimension, level)` pair (`levels_per_dim`=2 for BSQ/LFQ bits,
    `cfg.L` for FSQ), summed across the `dq` dimensions
    (`Σᵢ table[i, value_i]`), instead of one shared linear direction per
    dimension. Output format is unchanged from the continuous mode (still
    per-dim logits) — this only swaps the input. Strictly more expressive
    than the linear projection for binary dims (linear forces the two bit
    values' contributions to be exact negations of one shared direction;
    PQ lets them be fully independent), and — like the linear projection,
    for the same reason: both are sums of independent per-dimension
    terms — generalizes to any unseen *combination* of already-seen
    per-dimension values with zero extra training, unlike the vocab-table
    mode's OOV problem. Not yet combined with a full training run at
    session's end; implemented and smoke-tested.
- **FSQ path — unchanged, loosely coupled, interface Option A** (handover
  §2.1): `QCuteLM.forward()`'s FSQ branch decodes each chunk from the
  *encoder's own* code (`rec_loss`) and separately trains the LM to predict
  the next code (`pred_loss`) — the two objectives never interact.
  `QCuteLM.generate()`'s FSQ branch samples a level per dim from the LM's
  categorical distribution and decodes it directly — no re-encoding
  (deliberately not A-grounded).
- **BSQ path — tightly coupled by default** (`_forward_bsq_tightly_coupled`):
  `encoder → z_t → LM → raw latent → bsq_quantize → decoder → bytes_{t+1}`,
  graded against the *true* `bytes_{t+1}` — the decoder's primary
  reconstruction target is the LM's *prediction* of the next code, not the
  encoder's own code for that chunk. `pred_loss` (BCE between the LM's raw
  output and the true next code's sign bits) is kept alongside as a
  code-level supervision signal, independent of whether decoding it
  currently produces correct bytes. `cfg.aux_recon` (`--disable_aux_recon`
  to turn off) optionally adds back the *old* loosely-coupled term —
  `decoder(z_t)` reconstructing `bytes_t` directly, bypassing the LM — as
  an auxiliary regularizer; excluded from the reported `bpb_total` either
  way, since it's not part of the actual generative path (`generate()`
  never decodes the encoder's code directly). `QCuteLM.generate()`'s BSQ
  branch quantizes the LM's raw output the same deterministic way as
  training (no temperature — BSQ quantization is a sign, not a
  distribution to sample from; only FSQ's categorical sampling uses
  `temperature`).
  - **Results so far**: many BSQ variants tried (no-aux, plain-BSQ+aux,
    LFQ+aux, +uncertainty weighting, +entropy regularization, +AE
    pretraining, `dq` sweeps) — see status.md's Phase 2 for the full
    comparison table and per-run analysis. Best so far: LFQ+aux+uncertainty
    weighting, val_bpb 5.39 (beats the loosely-coupled architecture's 5.54,
    though with an open caveat about whether the mechanism or run variance
    is responsible). Consistent theme across variants: an LM-predictability
    vs. decoder-quality tradeoff — no variant has had both a well-predicted
    LM and a well-decoded byte output at once, until the MaskGIT decoder
    change (see status.md), which directly targets the decoder side.
- `QCuteLM.forward()` returns `rec_loss + pred_loss` (+ `aux_rec_loss` if
  enabled, BSQ only) — for FSQ this matches handover §7.1's training-step
  pseudocode (weight 1 each, quantized bottlenecks use `β=1` throughout
  per §1.5, no KL warmup needed); BSQ's tightly-coupled version is this
  repo's own architectural choice, not the handover doc's design. This is
  the *default* (unweighted) loss — `main()`'s training loop can instead
  recombine the raw per-term losses in `forward()`'s returned `metrics`
  dict via `--uncertainty_weighting` (learned per-loss `log_var` scalars,
  Kendall & Gal 2018), `--entropy_reg_weight` (adds `bsq_entropy_reg` on the
  LM's predicted latent), and/or `--disable_pred_loss` (drop `pred_loss`
  entirely — `rec_loss`'s gradient still reaches the LM via `bsq_quantize`'s
  STE, just without the explicit "match the true next code's bits"
  constraint; `pred_loss`/`latent_acc` are still computed/logged, just
  excluded from backward — an ablation testing whether that direct
  code-level supervision helps or fights the reconstruction-driven
  gradient). All three live in the training loop, not on `QCuteLM`, since
  loss-combination is a training choice, not architecture.
- **Optimizer**: all three of `qcutelm.py`'s `AdamW` instances (pretrain,
  vocab-LM, main joint loop) now use `betas=(0.9, 0.95), weight_decay=0.1`,
  matching `qcute/bytelm.py`/`qcute/bpelm.py` — previously they used plain
  PyTorch defaults (`betas=(0.9,0.999)`, `weight_decay=0.01`), an
  unexplained inconsistency across the three modules, now fixed.
- **Frozen-tokenizer alternative pipeline** (`--freeze_after_pretrain`,
  requires `--pretrain_ae`): the "dumber, simpler" alternative to joint
  tightly-coupled training — pretrain encoder+decoder to convergence,
  **freeze** them (the missing step vs. plain `--pretrain_ae`, which let
  joint training erase the pretrained solution), then train a **separate,
  plain categorical `LatentLM`** in discrete-vocabulary mode. `build_code_vocab`
  encodes the whole corpus with the frozen encoder and collects the
  distinct codes that actually occur (deduped on `targets`, the exact
  discrete code — not `z_hat`, unsafe to hash as floats) into a vocabulary,
  the same way BPE builds one from merges instead of embedding the full
  combinatorial codebook. `encode_to_vocab_ids` maps chunks to vocab ids
  (UNK for codes unseen in train — a real cost, e.g. ~17.6% UNK rate on
  val for one tested tokenizer). `train_vocab_lm` trains the vocab-mode
  `LatentLM` on the resulting id sequences, with a qualitative sample
  logged every eval so training progress is visible as actual text, not
  just bpb. Matches how FSQ/LFQ/BSQ papers train their downstream priors
  in the literature (frozen tokenizer, then a separate categorical model)
  — see status.md's literature-contrast discussion. Results so far:
  tokenizer pretrain plateaus well short of very high fidelity (68-85%
  `recon_acc` depending on mixer/LR/budget), and the downstream vocab-LM
  phase overfits fast on this tiny corpus (train `bpb`→~0.05 while val
  `bpb` climbs) — see status.md for the full trail.
- `split_train_val()` / `eval_metrics()`: same pattern as `qcute/bytelm.py` — a
  held-out val split with periodic evaluation of all training metrics
  (recon/latent accuracy, bpb_total, bpb_lm_only, plus aux terms when
  `aux_recon` is on).
- `score_continuation_bpb()`: mirrors whichever of the two `forward()`
  branches above applies (FSQ: decode-own-code; BSQ: decode-LM-prediction),
  restricted to the continuation-region chunks — the two-term bpb decomposition from
  `QCuteLM.forward()`, just sliced to a sub-range instead of averaged
  over the whole sequence.

Still independent of `qcute/bytelm.py` (no shared imports, including RoPE/attention
code, which is duplicated rather than factored out) — see the top of this
file for why.

## `qcute/bpelm.py` — BPE baseline

Handover §1.6 names BPE+MTP as the strong baseline; `qcute/bytelm.py` is
byte+MTP, so this module is the BPE half in isolation — a plain causal
transformer (same trunk as bytelm: pre-norm, RoPE, weight-tied head,
GPT-2-style init) over BPE token ids, **no MTP head** (bandwidth comes
purely from BPE merging here, by explicit choice, not stacked). Requires a
tokenizer trained via `scripts/train_bpe.py` (sentencepiece, BPE mode)
first — `--vocab_size` (default 8192, power-of-2) targets ~4 bytes/timestep
to match bytelm/qcutelm's tiny-corpus-scale default, reaching ~3.3
bytes/token on the tiny corpus (see the script's docstring: larger vocabs
on a 450KB corpus start memorizing phrases instead of generalizing — a
corpus-size ceiling, not a config problem; and even at full-corpus scale,
8 bytes/token is optimistic for natural-language BPE — typical scaling
tops out closer to 5-6, not 8).

- **Lossless tokenizer, not sentencepiece's NLP-oriented defaults.**
  `scripts/train_bpe.py` trains with `normalization_rule_name="identity"`,
  `remove_extra_whitespaces=False`, `byte_fallback=True`, and
  `add_dummy_prefix=False` — sentencepiece's normal defaults (NFKC
  normalization, whitespace collapsing, a synthetic leading space) silently
  drop information (verified empirically: encode→decode roundtrip changed
  `\n` to `" "` and lost bytes), which would make any downstream bpb claim
  false, not just approximate. The training script asserts a roundtrip
  check (`sp.decode(sp.encode(text)) == text`) and hard-fails if it doesn't
  hold — a lossy tokenizer isn't a `bpelm` config problem, it's a "don't use
  this .model file" problem.
- `build_byte_len_table()` / `bits_per_byte()`: BPB here isn't
  `mean_token_nats / avg_bytes_per_token` (biased, since common tokens tend
  to be short and rare tokens long) — it's the exact
  `sum(token_nats) / sum(that_token's_real_utf8_byte_length)`, computed via
  a precomputed per-token-id byte-length lookup table, **verified to sum to
  the exact original corpus byte count** (byte-fallback pieces like
  `<0x0A>` need special-casing to 1 byte each — their literal 6-character
  string form would otherwise be miscounted). This is what makes bpelm's
  bpb genuinely comparable to bytelm's and qcutelm's, not an estimate.
- Same `--config`, `Checkpointer`, `--eval_only`, and `lr_at` schedule as
  the other two, for a fair three-way comparison. No qualitative-generation
  or speculative-decoding support yet (narrower scope than bytelm/qcutelm
  — extend if needed).

## Data

`scripts/prepare_data.py` downloads `datasets/enwik8.gz` (~35MB, skipped if
present) and cuts `datasets/enwik8_1M.gz` (a 1,000,000-byte prefix, gzip'd,
`--subset_bytes` to change the size) — the standard corpus all three
training modules default `--data` to, so no `--n_bytes` cutoff is needed
for normal runs. `scripts/train_bpe.py` trains the sentencepiece tokenizer
`qcute.bpelm` needs, from either file.

## Configs

`configs/` holds named, reproducible experiments as plain Python files (see
the config-file bullet above), each with a docstring documenting what it
reproduces, the measured result, and the exact run/plot commands:
`bytelm_xs_mtp4.py` (`xs` preset, 2000 steps, best val_bpb 2.52),
`bpelm_8192.py` (vocab=8192, 2000 steps, best val_bpb 2.35),
`qcutelm_bsq_k4_frozen_vocab.py` (K=4, tightly-coupled BSQ, 2-layer
tokenizer, `--pretrain_ae` to 95% then `--freeze_after_pretrain` + plain
vocab-LM training — the closest architectural match to `bpelm_8192.py`:
both are a categorical softmax LM over a fixed discrete vocabulary, cross-
entropy only, no decoder-side loss during the LM phase). The many further
qcutelm variants tried (LFQ, uncertainty weighting, entropy regularization,
MaskGIT decoder, `dq` sweeps, unfrozen joint training with aux recon — see
status.md's Phase 2 for the full list and results) were run via direct CLI
invocation, not saved as config files yet. Add a new file here rather than
a long inline CLI invocation whenever a run is worth being able to
reproduce by name later.

## Known gaps vs. the full design (tracked in [status.md](status.md))

- Encoder/decoder are non-streaming chunk-local (attention- or conv-mixed,
  no longer plain MLPs — see the mixer-bug-fix entry above), not the doc's
  recommended causal-SSM encoder / streaming-SSM decoder (handover §1.3,
  §1.4.1) — chunk-boundary context is not shared across chunks.
- Interface is Option A (cheapest, most drift-prone); the doc's recommended
  default is A-grounded (handover §2.4).
- Decoder training is MaskGIT-style masked CE (handover §1.4b, added this
  session), not the doc's recommended time-free masked diffusion (§1.4.3c)
  — MaskGIT gives a heuristic training loss ("strong empirically" per the
  doc), not a proper BPB ELBO; only the diffusion variant does.
- No geometric-state mixers (Phase 3) — both `qcute/bytelm.py` and
  `qcute/qcutelm.py`'s attention is plain softmax.
- BSQ's entropy regularization (`bsq_entropy_reg`, a real gap vs. the LFQ/
  BSQ papers, added this session) destabilizes training when combined with
  uncertainty weighting (see status.md) — current wiring isn't right yet,
  needs a fixed small coefficient or warmup instead of free adaptive
  weighting.
- The vocab-table LM mode (`train_vocab_lm`) has a real OOV/UNK problem
  (codes unseen during vocab-building get no meaningful representation) —
  `--lm_factorized_input` (`FactorizedCodeEmbedding`) is built specifically
  to remove this, but not yet run end-to-end as a full training comparison.
- `bpb_total`/`bpb_lm_only` aren't directly comparable across
  `--disable_pred_loss` on vs. off: `bpb_pred` is derived straight from the
  (possibly uncalibrated, when `pred_loss` isn't trained) raw BCE value,
  not from actual decode quality — a `--disable_pred_loss` run can show a
  much worse `bpb_total` than one with `pred_loss` on even when
  `recon_acc` is actually better, since nothing calibrates `v_pred`'s
  magnitude/probabilities without the BCE term. Compare `recon_acc`
  directly across such runs, not `bpb_total`.
