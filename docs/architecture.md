# Architecture

Two self-contained modules in the `qcute/` package: `qcute/bytelm.py` and
`qcute/qcutelm.py`. Neither imports the other or shares a common name with
the package itself (an earlier top-level `qcute.py` next to the `qcute/`
package collided on `import qcute` — renamed; `qcute/bytelm.py` inlines its own
tiny `load_enwik8` rather than importing it from `qcute/qcutelm.py`, to
keep the two independent). Merge them once a third module needs to reuse
encoder/decoder/LM pieces.

See [continuous_tokenizer_handover.md](continuous_tokenizer_handover.md) for
the full design this implements, and [status.md](status.md) for what's done.

Both scripts' `main()` follow the same (duplicated, not shared — see
docstrings for the extraction candidates identified but not yet acted on)
conventions:

- **Logging**: a `tqdm` progress bar for live terminal feedback; at
  `--log_every`/`--eval_every`, `Logger` writes the same line to *both*
  `logs/<name>.log` (raw text, `tail -f` from another terminal) and
  `logs/<name>.jsonl` (structured, one record per line, prefixed with
  elapsed `[HH:MM:SS]` / `elapsed_s` rather than a raw epoch timestamp).
  `<name>` resolves as `--log_file` (explicit) → `--config`'s filename stem
  → a script-specific default formula. `logs/` is gitignored.
- **Config files**: `--config path/to/file.py` loads a plain Python module
  (see `configs/`) whose module-level variables become new argparse
  defaults (`load_config_module` + `parser.set_defaults(...)`, filtered to
  known flags) — **CLI flags still override config file values**, which
  override the script's hardcoded defaults. Two-pass parse: a small
  pre-parser reads just `--config` before the full parser is built.
- **Checkpointing**: `Checkpointer` saves `checkpoints/<name>_best.pt`
  (overwritten only when the tracked val metric improves) and
  `checkpoints/<name>_last.pt` (overwritten every `--save_every_n_evals`
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
t+1..t+n from one trunk pass, bandwidth-matched to `qcute.qcutelm`'s
default `K=8` (`mtp_heads` defaults to 8) so BPB and generation latency are
comparable at matched bandwidth between the two scripts.

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

## Data

`scripts/prepare_data.py` downloads `datasets/enwik8.gz` (~35MB, skipped if
present) and cuts `datasets/enwik8_tiny.gz` (a 500,000-byte prefix, gzip'd,
`--tiny_bytes` to change the size) for fast local/smoke runs — both
`qcute.bytelm` and `qcute.qcutelm` accept either via `--data`.

## Configs

`configs/` holds named, reproducible experiments as plain Python files (see
the config-file bullet above) — e.g. `bytelm_xs_tiny_longrun.py` (the
double-descent exploration run: `xs` preset, tiny dataset, 25000 steps) and
`qcutelm_bsq_tiny.py` (matched-bandwidth BSQ companion). Add a new file here
rather than a long inline CLI invocation whenever a run is worth being able
to reproduce by name later.

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
