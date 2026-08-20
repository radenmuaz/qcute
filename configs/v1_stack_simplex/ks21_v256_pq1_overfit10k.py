"""v1_stack_simplex/ks21_v256_pq1_overfit10k: first-ever qcute_v1 training run -- validates the stage-1
autoencoder-decode rewrite (see docs/qcute_v1_plan.md). Ks=(2,1): level0 decode is now a pure
autoencoder (code_window=1, reconstructs its own 2-byte block from its own code alone, no
cross-attention to any other level); level1 (top) is unchanged genuine NTP over codes.
quant_type=simplex, vocab=256 (single 256-way softmax, pq_chunks=1, 8 bits) -- the "no PQ"
baseline paired against ks21_v64_pq4.py's PQ variant (64-way x4 chunks, 24 bits combinatorial),
testing whether more independent low-resolution chunks beats one high-resolution softmax here
too (the same pattern the v5 FSQ ablation already found, see docs/status.md).
Follows CLAUDE.md's standing overfit10k fast-iteration methodology (n_bytes=10000, context=256,
steps=1000) since this is an unvalidated new architecture, not a full-scale run.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks21_v256_pq1_overfit10k.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks21_v256_pq1_overfit10k
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks21_v256_pq1_overfit10k"
decoder_type = "stack"
Ks = (2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 256
pq_chunks = 1
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 1000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 64  # StackDecoderV1's own check_roundtrip_consistency/check_decode_modes ARE
# implemented for this decoder_type (unlike the newer stack_v2/stack_v2_local, whose generation
# path is still unsettled, see chat/docs 2026-08-20) -- overfit sanity check: can it memorize and
# regenerate the training corpus via decode-from-own-code round-trip?
