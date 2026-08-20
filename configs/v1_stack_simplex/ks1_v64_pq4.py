"""v1_stack_simplex/ks1_v64_pq4: full-scale (full enwik8_1M) run, following the overfit10k
validation in ks1_v64_pq4_overfit10k.py. n_levels=1, vocab=64/pq_chunks=4 PQ variant paired
against ks1_v256_pq1.py's single-softmax baseline. See ks1_v256_pq1.py for scale/schedule notes.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks1_v64_pq4.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks1_v64_pq4
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks1_v64_pq4"
decoder_type = "stack"
Ks = (1,)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 64
pq_chunks = 4
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

qual_gen_bytes = 0
