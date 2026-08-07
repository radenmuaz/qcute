"""qcute.qcutelm_vlt6 config: FLOP-matched (not just context-matched) vs
bpelm_8192. FLOP proxy = d_model^2 * n_layers * tokens_processed:
  bpelm_8192:  256^2 * 4 * 256           = 67,108,864
  this config: 64^2*2*2048 (enc/dec) + 256^2*3*256 (codelm)
             = 16,777,216 + 50,331,648   = 67,108,864  <- exact match

context_len=1024 (K=4) still gives CodeLM 256 codes, matching bpelm's
256-token context (see qcutelm_vlt6_ifsq_vs_bpelm.py). Encoder/decoder
kept cheap (d_model=64, n_layers=2, 0.133M params — tokenizer overhead is
real cost here, unlike bpelm's non-parametric BPE tokenizer). CodeLM
d_model=256 but n_layers=3 (not 4) — the layer count that makes the FLOP
proxy land exactly on bpelm's, given enc/dec's own (fixed, small) share.
Total params 2.529M (smaller than bpelm's ~3.7M, since this matches
compute not parameter count — can't hit both simultaneously, see session
notes).

    uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_ifsq_flopmatch_bpelm.py
"""
from pathlib import Path

K = 4
context_len = 1024
attn_window = 16
dq = 6
quant_type = "ifsq"
fsq_levels = 8

d_model = 64
n_heads = 4
n_layers = 2
mlp_mult = 4
code_net_layers = 0

lm_d_model = 256
lm_n_heads = 4
lm_n_layers = 3
lm_mlp_mult = 4

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

gen_every = 200
gen_prompt_len = 64
gen_new_bytes = 64
