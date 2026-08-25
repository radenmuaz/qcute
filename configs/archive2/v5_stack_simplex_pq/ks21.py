"""v5_stack_simplex_pq/ks21: same as qcute_v5_stack_noreg/ks21.py (Ks=(2,1)) but
quant_type="simplex" with pq_chunks=32/vocab=8 -- product-quantized categorical code, standard
PQ convention (vocab=8 is the FIXED per-chunk codebook size, pq_chunks=32 multiplies it):
total code width 8*32=256 (same head width as the flat vocab=256/pq_chunks=1 baseline, same
param count), combinatorial capacity 8^32 instead of a flat 256. vocab=8/pq_chunks matches the
FSQ ablation's empirical sweet spot (grid_levels=8 per unit, see docs/status.md).

uv run python -m qcute.qcute_v5 --decoder_type stack --config configs/v5_stack_simplex_pq/ks21.py

# plot after training:
uv run python scripts/plot_run.py logs/v5_stack_simplex_pq_ks21
"""
from pathlib import Path

run_name = "v5_stack_simplex_pq_ks21"
decoder_type = "stack"
Ks = (2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 8
pq_chunks = 32
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
