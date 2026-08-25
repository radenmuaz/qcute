"""v5_stack_gmmdiag_pq/ks21: same as qcute_v5_stack_noreg/ks21.py (Ks=(2,1)) but
quant_type="gmm_diag" with gmm_dq=4/pq_chunks=4/gmm_k=8 -- product-quantized GMM-diag, standard
PQ convention (gmm_k=8 is the FIXED per-chunk codebook size, dq=4 is divided into pq_chunks=4
groups of dq_sub=1 each -- fully per-dimension, gmm_dq/pq_chunks kept SMALL/HIGH respectively
since gmm_diag has been the slowest quant_type observed this session, ~4h+ for a single
Ks=(1,) run vs ~30-65min for grid/simplex at comparable scale). Capacity 8^4=4096, an exact
apples-to-apples match against v5_stack_fsq/ks1.py's grid_dq=4/grid_levels=8 (same 4096
capacity) -- isolates whether GMM-diag's LEARNED per-dim Gaussian bins beat FSQ's fixed
uniform grid at identical combinatorial capacity.

uv run python -m qcute.qcute_v5 --decoder_type stack --config configs/v5_stack_gmmdiag_pq/ks21.py

# plot after training:
uv run python scripts/plot_run.py logs/v5_stack_gmmdiag_pq_ks21
"""
from pathlib import Path

run_name = "v5_stack_gmmdiag_pq_ks21"
decoder_type = "stack"
Ks = (2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_hard = True
code_sample = False
quant_type = "gmm_diag"
gmm_dq = 4
gmm_k = 8
pq_chunks = 4
vocab = 256
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
