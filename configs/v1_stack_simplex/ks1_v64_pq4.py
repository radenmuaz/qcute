"""v1_stack_simplex/ks1_v64_pq4: same as ks1_v256_pq1.py but vocab=64 (per-chunk width),
pq_chunks=4 -- 4 independent 64-way softmaxes instead of one 256-way softmax, same total code
width. n_levels=1, so decode is the unchanged-from-v5 top-level NTP path (see ks1_v256_pq1.py's
docstring); this config isolates whether the PQ-vs-no-PQ pattern from v5's FSQ/simplex-PQ
ablations still holds at qcute_v1's baseline (n_levels=1) scale.

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

qual_gen_bytes = 0
