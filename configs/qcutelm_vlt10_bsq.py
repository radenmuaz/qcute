"""qcute.qcutelm_vlt10 config: Clockwork-RNN-inspired 2-level sandwich
(lower tokenizer layer -> codelm as sparse middle layer -> upper tokenizer
layer). attn_window=64 for both tokenizer tiers (context_len=1024 -> 16
chunks) and lm_attn_window=64 for codelm (n_blocks=256 -> 4 chunks), per
the design's "attention window 64 for both tokenizer and codelm" (memory
savings; codelm's window still covers more raw-byte context per chunk than
the tokenizer tiers' since each of its tokens already = K=4 bytes). Same
K/dq/quant_type/d_model/lm_d_model as the qcutelm_vlt7/vlt8 bsq runs for
direct comparability.

    uv run python -m qcute.qcutelm_vlt10 --config configs/qcutelm_vlt10_bsq.py
"""
from pathlib import Path

K = 4
context_len = 1024
dq = 13
quant_type = "bsq"
fsq_levels = 8

d_model = 96
n_heads = 4
n_layers = 2
mlp_mult = 4
attn_window = 64

lm_d_model = 256
lm_n_heads = 4
lm_n_layers = 3
lm_mlp_mult = 4
lm_attn_window = 64

code_match_weight = 1.0

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
weight_decay = 1e-5
warmup_steps = 500
cosine_decay = False
constant_steps = 100
eval_every = 100
eval_batches = 20

gen_every = 1000
gen_prompt_len = 64
gen_new_bytes = 64
