"""qcute.qcutelm_vlt4 first experiment: strided-readout tokenizer (fork of
qcute.qcutelm_vlt3 — see that module's docstring for the design: instead
of a trainable code-query token limited to a K-byte window, the shared
Block stack trains as a regular large-context causal LM, and codes are
read off every K-th hidden state by a small code_net).

context_len=128 (32 K=4 blocks per training example) gives the LM real
context to work with, unlike qcutelm_vlt3's context-free K-byte windows.
Same d_model/n_layers/schedule as qcutelm_vlt3_4layer.py for a direct
comparison; code_net_layers=0 (plain linear readout) as the simplest
starting point.

    uv run python -m qcute.qcutelm_vlt4 --config configs/qcutelm_vlt4_ctx128.py
"""
from pathlib import Path

K = 4
context_len = 128
attn_window = 128  # matches context_len — see qcutelm_vlt4.py's ZeroKVCausalSelfAttention docstring
dq = 18
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
