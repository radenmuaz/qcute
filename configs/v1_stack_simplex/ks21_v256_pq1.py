"""v1_stack_simplex/ks21_v256_pq1: full-scale (full enwik8_1M) run, following the overfit10k
validation in ks21_v256_pq1_overfit10k.py. Ks=(2,1): level0 decode is the BOS-interleaved
self-attn + cross-attn-to-own-code mechanism (see docs/qcute_v1_plan.md); level1 (top) is
unchanged genuine NTP over codes. quant_type=simplex, vocab=256, pq_chunks=1 -- paired against
ks21_v64_pq4.py's PQ variant. Scale/schedule matches the v5_stack_fsq/* full-scale convention.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks21_v256_pq1.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks21_v256_pq1
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks21_v256_pq1"
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
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 200
eval_every = 2000
eval_batches = 20
full_val_eval = True

qual_gen_bytes = 128
qual_prompt_bytes = 64
