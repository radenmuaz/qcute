"""qcute.qcutelm_vlt5 first experiment: continuous (non-reset) reconstruction
tokenizer, BSQ. Based off qcutelm_vlt4_ctx128.py (same d_model/n_layers/
schedule) but attn_window=16 (sliding window, not full causal) and the
redesigned continuous reconstruction path — see qcutelm_vlt5.py's module
docstring for the worked example (c4,1,2,3,c8,5,6,7 -> 1,2,3,4,5,6,7,8).

Focus: get reconstruction match to 95% first (recon_target_acc), pretraining
only — no joint latent-LM here (that's qcutelm_vlt4's --joint_lm, to be
revisited once this fork's reconstruction quality is established).

    uv run python -m qcute.qcutelm_vlt5 --config configs/qcutelm_vlt5_bsq.py
"""
from pathlib import Path

K = 4
context_len = 512  # longer seq/step (like bytelm's context) for training throughput; attn_window stays 16
attn_window = 16
dq = 18
quant_type = "bsq"
d_model = 128
n_heads = 4
n_layers = 4
mlp_mult = 4
code_net_layers = 0
ntp_loss_weight = 1.0
recon_loss_weight = 1.0

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 100000
batch_size = 16
lr_peak = 6e-4
weight_decay = 1e-5
warmup_steps = 500
cosine_decay = True
constant_steps = 100
eval_every = 100
eval_batches = 20

recon_target_acc = 0.95
