"""v5_stack_gmm/ks1: Ks=(1,), stack decoder, quant_type="gmm" (full-covariance shared GMM
codebook -- precision-Cholesky NLL, manual-triangular-solve reparam sampling), gmm_k=8/gmm_dq=4 --
same dq=4/component-count=8 sizing as configs/v5_stack_fsq/ks1.py's grid_dq=4/grid_levels=8 (dq
matched exactly; component count K mirrors FSQ's per-dim level count rather than FSQ's full
combinatorial L^dq code space, since K is a literal codebook-entry count for GMM, not combinatorial).
Same architecture/schedule otherwise (code_hard=True/code_sample=False, entropy_reg_weight=0.0,
full val eval).

uv run python -m qcute.qcute_v5 --decoder_type stack --config configs/v5_stack_gmm/ks1.py

# plot after training:
uv run python scripts/plot_run.py logs/v5_stack_gmm_ks1
"""
from pathlib import Path

run_name = "v5_stack_gmm_ks1"
decoder_type = "stack"
Ks = (1,)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_hard = True
code_sample = False
quant_type = "gmm"
gmm_k = 8
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
