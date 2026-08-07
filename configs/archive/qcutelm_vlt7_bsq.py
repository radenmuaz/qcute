"""qcute.qcutelm_vlt7 config: BSQ quantizer, matching qcutelm_vlt6 grid's
own bsq scale (dq=13, exact bpelm-vocab match: 2^13=8192) and narrow-
tokenizer/wide-codelm split (d_model=96/n_layers=2 tokenizer,
lm_d_model=256/n_layers=3 codelm), context_len=1024 (256 code-level
positions, matching bpelm's 256-token context) — same architecture scale
as qcutelm_vlt6_grid_bsq_ntp_encdec.py (best qcute result so far this
session, val_bpb 2.5872), so this is a direct, fair first real-data test
of the v3 hybrid design (symmetric narrow tokenizer + separate wide
codelm forecasting over the short code sequence) against that number.

    uv run python -m qcute.qcutelm_vlt7 --config configs/qcutelm_vlt7_bsq.py
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
attn_window = 64  # tokenizer tier: O(T*64) chunked instead of O(T^2) dense — Pass1/Pass2 sequence length is
                   # n_blocks*(K+1)=1280, divides evenly by 64. Matches qcutelm_vlt6 grid cells' own attn_window.

lm_d_model = 256
lm_n_heads = 4
lm_n_layers = 3
lm_mlp_mult = 4

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
