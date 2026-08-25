"""v5_stack_gmm/ks1_256: same as ks1.py (Ks=(1,), quant_type="gmm", full covariance) but
gmm_k=256 (vs ks1.py's gmm_k=8), gmm_dq=4 unchanged -- sized to match SimplexQuant's 256-way
vocab as an apples-to-apples reference point rather than gmm_k=8192, which blew up the head
(GMM's gating head width scales linearly with K, unlike FSQ/BSQ whose head is decoupled from
their combinatorial code count -- K=8192 meant a 256->122880 linear per head, ~63M params for
just the two heads). K=256 keeps head width at 256*(1+4+10)=3840, ~2M params total, in line with
the rest of the model.

uv run python -m qcute.qcute_v5 --decoder_type stack --config configs/v5_stack_gmm/ks1_256.py

# plot after training:
uv run python scripts/plot_run.py logs/v5_stack_gmm_ks1_256
"""
from pathlib import Path

run_name = "v5_stack_gmm_ks1_256"
decoder_type = "stack"
Ks = (1,)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_hard = True
code_sample = False
quant_type = "gmm"
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
