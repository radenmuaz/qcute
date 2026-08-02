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
bytes/token on a 450,000-byte corpus before larger vocabs start
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
  `2^18`). Both straight-through; `FSQ.forward` and `BSQ.forward` return
  both the STE-quantized `z_hat` (fed to the LM/decoder) and the discrete
  targets (levels for FSQ, bits for BSQ) the LM head is trained against.
- `ChunkEncoder`: byte-embed the K bytes, flatten, 2-layer MLP → bottleneck.
  `ChunkDecoder`: mirror MLP, `z → K*vocab` logits, one-shot factorized CE
  (handover §1.4.3a) — no MaskGIT masking.
- `LatentLM`: causal transformer over the code sequence, RoPE on Q/K
  (duplicated from `qcute/bytelm.py` rather than imported — see below), input is
  a linear projection of `z_hat` (continuous), output is per-dim categorical
  logits (FSQ) or per-dim bit logits (BSQ). This is interface **Option A**,
  pure latent autoregression (handover §2.1) — deliberately *not*
  A-grounded: `QCuteLM.generate()` feeds the LM's sampled code straight
  back as the next input, no re-encoding of decoded bytes. Same GPT-2-style
  init as `qcute/bytelm.py` (load-bearing for the same reason — see below).
- `QCuteLM.forward()` returns `rec_loss + pred_loss` (handover §7.1
  training-step pseudocode): reconstruction CE from the decoder plus
  next-code CE/BCE from the LM, both weight 1 (quantized bottlenecks use
  `β=1` throughout per §1.5, no KL warmup needed).
- `QCuteLM.generate(prompt_chunks, n_chunks)`: encodes a byte prompt,
  then autoregressively samples the next code from the LM (categorical for
  FSQ, Bernoulli for BSQ) and decodes it to bytes each step (handover §7.2,
  Option-A variant — no `z_grounded = Encoder(bytes)` re-encoding step).
- `split_train_val()` / `eval_metrics()`: same pattern as `qcute/bytelm.py` — a
  held-out val split with periodic evaluation of all training metrics
  (recon/latent accuracy, bpb_total, bpb_lm_only).
- `score_continuation_bpb()`: unlike bytelm's single causal forward, this
  re-runs encoder→bottleneck→LM→decoder on prompt+continuation chunks and
  restricts both the reconstruction CE and the next-code CE/BCE to
  continuation-region chunks — the two-term bpb decomposition from
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
present) and cuts `datasets/enwik8_tiny.gz` (a 500,000-byte prefix, gzip'd,
`--tiny_bytes` to change the size) for fast local/smoke runs.
`scripts/train_bpe.py` trains the sentencepiece tokenizer `qcute.bpelm`
needs, from either. All three training modules accept `--data`.

## Configs

`configs/` holds named, reproducible experiments as plain Python files (see
the config-file bullet above) — e.g. `bytelm_xs_tiny_longrun.py` (the
double-descent exploration run: `xs` preset, tiny dataset, 25000 steps),
`qcutelm_bsq_tiny.py` (matched-bandwidth BSQ companion), and
`bpelm_tiny.py` (the BPE companion — needs `datasets/bpe_enwik8_tiny_8192.model`
from `scripts/train_bpe.py` first). Add a new file here rather than a long
inline CLI invocation whenever a run is worth being able to reproduce by
name later.

## Known gaps vs. the full design (tracked in [status.md](status.md))

- Encoder/decoder are non-streaming chunk-local MLPs, not the doc's
  recommended causal-SSM encoder / streaming-SSM decoder (handover §1.3,
  §1.4.1) — chunk-boundary context is not shared across chunks.
- Interface is Option A (cheapest, most drift-prone); the doc's recommended
  default is A-grounded (handover §2.4).
- Decoder training is one-shot factorized, not the recommended time-free
  masked diffusion (handover §1.4.3c) for a proper BPB ELBO.
- No geometric-state mixers (Phase 3) — both `qcute/bytelm.py` and
  `qcute/qcutelm.py`'s attention is plain softmax.
