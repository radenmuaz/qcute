# qcute

("Quantized Continuous Tokenizer") — continuous byte-level tokenizer + LM.
**Current active designs: `qcute/qcute_v5_concat.py` and
`qcute/qcute_v5_stack.py`** — recursive NTP tower, `LevelLM` per level,
cross-level fusion instead of a separate decoder, forked from the
now-archived `qcute_refine` lineage's `v4_4_1.py`/`v4_5_1.py` (see
[CLAUDE.md](CLAUDE.md)'s own Architecture section for the full lineage
summary, and [docs/status.md](docs/status.md) for session-by-session
results). Math for the LM-continuation decode mechanism both share:
[docs/qcute_refine_v4_4_1_v4_5_1_math.md](docs/qcute_refine_v4_4_1_v4_5_1_math.md)
(math for the original v1 design, now archived:
[docs/archive2/qcute_refine_math.md](docs/archive2/qcute_refine_math.md)).
The original design spec this project started from,
[docs/archive/continuous_tokenizer_handover.md](docs/archive/continuous_tokenizer_handover.md),
now describes an archived earlier lineage (`qcute/archive/`, including
`qcutelm.py`) superseded by `qcute_refine`, itself now archived under
`qcute/archive2/` and superseded by `qcute_v5_concat`/`qcute_v5_stack` —
kept for historical reference. See [docs/architecture.md](docs/architecture.md)
for how the still-active baselines (`bytelm`/`bpelm`) map to their own
design, and [docs/status.md](docs/status.md) for phase-by-phase progress.

## Quickstart

```bash
uv sync

# downloads datasets/enwik8.gz (~35MB) and cuts datasets/enwik8_1M.gz
# (1,000,000-byte prefix, for fast smoke/local runs)
uv run python scripts/prepare_data.py

# byte-level baseline LM w/ MTP head, reports bits-per-byte
uv run python -m qcute.bytelm --preset sd
uv run python -m qcute.bytelm --preset xs --data datasets/enwik8_1M.gz  # quick local run

# BPE baseline — train the tokenizer first
uv run python scripts/train_bpe.py --data datasets/enwik8_1M.gz
uv run python -m qcute.bpelm --sp_model datasets/bpe_enwik8_1M_8192.model --data datasets/enwik8_1M.gz

# the active designs — recursive NTP tower + cross-level fusion
uv run python -m qcute.qcute_v5_concat --config configs/qcute_v5_concat_overfit10k_k4single.py
uv run python -m qcute.qcute_v5_stack --config configs/qcute_v5_stack_overfit10k_k4single.py

# or any named, reproducible config (CLI flags still override individual values) —
# EVERY file under configs/ has its own module docstring explaining what it tests
# and giving its exact `uv run` invocation; open the file if unsure what to run
uv run python -m qcute.bytelm --config configs/bytelm_xs_mtp4_ctx1024.py
uv run python -m qcute.bpelm --config configs/bpelm_8192.py

# evaluate a saved checkpoint only, no training
uv run python -m qcute.bytelm --eval_only --checkpoint_path checkpoints/<run_name>/best.pt --data datasets/enwik8_1M.gz
```

All modules run on CUDA/MPS/CPU automatically. **Only run one training job
at a time** — see [CLAUDE.md](CLAUDE.md) for why.

## So far

- `qcute/bytelm.py`: byte-level causal transformer with RoPE, `mtp_heads` parallel
  next-byte heads bandwidth-matched to `qcute.qcutelm`'s K (handover §1.6's
  BPE+MTP baseline, byte-level), train/val split + periodic eval, exact BPB
  from head 0. Presets `xs` (~3.7M, for quick local runs), `sd` (~101M),
  `md` (~403M). Also includes a self-speculative decoding generator (MTP
  heads as draft, verified against a true causal pass) to benchmark
  generation latency against `qcute.qcutelm`'s K-bytes-per-step decode.
- `qcute/qcute_v5_concat.py`/`qcute/qcute_v5_stack.py` (standalone
  modules, forked from the archived `qcute_refine` lineage's
  `v4_4_1.py`/`v4_5_1.py`): recursive NTP tower — each level (`LevelLM`)
  embeds its own input, runs causal self-attention, quantizes into a
  code for the level above every K positions (`Config.quant_type`:
  categorical softmax, default, or `"bsq"` binary spherical
  quantization), and cross-attends to the level-above's own hidden state
  before its own self-attention runs ("fusion") so its own next-token
  loss can depend on the coarser code — earlier `qcute_refine` versions
  (v1/v2) used a separate `DecoderLevel` for this instead, later found
  to do the same job at higher cost. Full architecture history and
  current results: [CLAUDE.md](CLAUDE.md), [docs/status.md](docs/status.md).
  `qcute/qcutelm.py` (the original encoder+FSQ/BSQ+latent-LM+decoder
  design this superseded) is now **archived** under `qcute/archive/` —
  still runnable via `qcute.archive.qcutelm` for historical reference,
  not part of active work; see
  [docs/archive/status_archive.md](docs/archive/status_archive.md) for
  its own trail of variants tried (LFQ vs. BSQ, MaskGIT decoder,
  uncertainty weighting, etc.) and [docs/architecture.md](docs/architecture.md)
  for its design.
- `qcute/bpelm.py`: a sentencepiece-BPE-tokenized causal transformer, same
  trunk shape as bytelm, with exact byte-weighted BPB (not the naive
  mean-tokens-per-avg-token-length approximation) so it's genuinely
  comparable to the other baselines. Needs `scripts/train_bpe.py` run
  first. Has `generate_ar`/`generate_kv_cache`/`validate_generation`
  functions, but no `--qual_gen_bytes` CLI flag of its own yet — narrower
  scope than bytelm.
- All modules support: a `--config <file.py>` (see `configs/` — every
  config file has its own module docstring explaining what it tests and
  its exact `uv run` invocation, copy-pasteable directly from the file)
  with CLI flags overriding individual config values; checkpointing
  (`checkpoints/`, gitignored) that keeps the best-so-far and
  most-recent model, plus `--eval_only --checkpoint_path ...` to evaluate
  without training. `qcute.bytelm` additionally supports
  `--qual_gen_bytes` for qualitative generation — a prompt (from
  train/val data or `--qual_user_text`) alongside the model's
  continuation, the real ground-truth continuation when available, and
  the model's bpb on it.
- All modules are self-contained (no shared internal submodules); see
  [docs/architecture.md](docs/architecture.md) for why and when to split further.
- Superseded design (streaming-causal-encoder standalone autoencoder,
  no LM) kept for reference in `archive/`.

Details, gaps, and next steps: [docs/status.md](docs/status.md). For how this
repo itself was scaffolded (reusable for future projects):
[docs/scaffolding_playbook.md](docs/scaffolding_playbook.md).
