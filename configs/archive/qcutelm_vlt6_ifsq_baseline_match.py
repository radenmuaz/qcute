"""qcute.qcutelm_vlt6 config: context_len=256 (matching bytelm_xs/bpelm_8192's
context exactly) with BOTH encoder/decoder and CodeLM following baseline
dims exactly (d_model=256, n_layers=4, n_heads=4, mlp_mult=4 — same as
bytelm_xs/bpelm_8192). Closest simple config to matching both params and
FLOPs simultaneously: params=6.458M (~1.75x bytelm's ~3.7M, since two
full-size stacks instead of one), FLOP-proxy token-ratio=576/256=2.25x
(encoder 256 + decode ~256 + codelm 256/K=64 token-positions per step vs
bytelm's single 256-token pass) — can't hit both exactly at once given
this architecture's 3-pass-per-step shape (see session notes), this is
the closest "just follow baseline" config.

    uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_ifsq_baseline_match.py
"""
from pathlib import Path

K = 4
context_len = 256
attn_window = 16
dq = 6
quant_type = "ifsq"
fsq_levels = 8

d_model = 256
n_heads = 4
n_layers = 4
mlp_mult = 4
code_net_layers = 0

lm_d_model = 256
lm_n_heads = 4
lm_n_layers = 4
lm_mlp_mult = 4

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

gen_every = 1000
gen_prompt_len = 64
gen_new_bytes = 64
