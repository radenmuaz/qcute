"""v5_stack_gmm/ks1_256_diag: same as ks1_256.py but quant_type="gmm_diag" (diagonal
covariance) -- sibling A/B, gmm_k=256/gmm_dq=4 unchanged. Cheaper head (256*(1+2*4)=2304 vs
full-cov's 3840) and no triangular-solve sampling path.

uv run python -m qcute.qcute_v5 --decoder_type stack --config configs/v5_stack_gmm/ks1_256_diag.py

# plot after training:
uv run python scripts/plot_run.py logs/v5_stack_gmm_ks1_256_diag
"""
from pathlib import Path

run_name = "v5_stack_gmm_ks1_256_diag"
decoder_type = "stack"
Ks = (1,)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_hard = True
code_sample = False
quant_type = "gmm_diag"
gmm_k = 256
gmm_dq = 4
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
