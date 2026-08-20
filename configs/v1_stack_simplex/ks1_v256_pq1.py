"""v1_stack_simplex/ks1_v256_pq1: full-scale (full enwik8_1M, ~1M bytes) run, following the
overfit10k validation in ks1_v256_pq1_overfit10k.py. n_levels=1: level0 IS top, so decode is the
unchanged-from-v5 genuine self-code-recurrent NTP path (see that config's docstring for why).
quant_type=simplex, vocab=256, pq_chunks=1 -- paired against ks1_v64_pq4.py's PQ variant.
Scale/schedule matches the v5_stack_fsq/* full-scale convention (steps=8000, context=256).

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack_v1 --config configs/v1_stack_simplex/ks1_v256_pq1.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks1_v256_pq1
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks1_v256_pq1"
decoder_type = "stack"
Ks = (1,)
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

qual_gen_bytes = 0
