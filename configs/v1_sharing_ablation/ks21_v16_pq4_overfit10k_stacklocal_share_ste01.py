"""v1_sharing_ablation/ks21_v16_pq4_overfit10k_stacklocal_share_ste01: 4 of 8 in the
weight-sharing ablation grid (2026-08-23) -- see ks21_v16_pq4_overfit10k_stack_noshare_ste01.py
for the full grid description and ks21_v16_pq4_overfit10k_stacklocal_noshare_ste01.py for why
this Ks has no tunable "upper code window" knob under stack_local.

This config: Ks=(2,1), decoder_type="stack_local", share (kv_lm_mode="shared",
decoder_own_stage_mode="shared"). byte_head_tied=False.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack_local --config configs/v1_sharing_ablation/ks21_v16_pq4_overfit10k_stacklocal_share_ste01.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_sharing_ablation_ks21_v16_pq4_overfit10k_stacklocal_share_ste01
"""
from pathlib import Path

run_name = "v1_sharing_ablation_ks21_v16_pq4_overfit10k_stacklocal_share_ste01"
decoder_type = "stack_local"
Ks = (2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 16
pq_chunks = 4
kv_lm_mode = "shared"
decoder_own_stage_mode = "shared"
byte_head_tied = False
encoder_ste_p = 0.1
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

steps = 3000

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 64
