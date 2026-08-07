"""qcute.qcutelm_vlt6 config: ~2x qcutelm_vlt6_ifsq_flopmatch_bpelm.py's
compute, but leaning on a wider attention window (attn_window=16->64,
real added compute, ZERO added params — window only changes how much of
the already-computed K/V each query attends to, not the weight count) to
"compensate less params" than a naive width/depth doubling would cost.

MLP-FLOP proxy (d^2*n_layers*tokens, the dominant param-costly term):
  naive 2x (d=128,n=2 enc/dec + d=256,n=4 codelm): 134.2M (2.00x bpelm), 3.664M params
  this config (d=96,n=2 enc/dec + d=256,n=3 codelm): 88.1M (1.31x bpelm) + extra
    real attention compute from the 4x-wider window (not captured in this proxy),
    2.678M params (27% fewer than the naive-2x option)

Still context_len=1024 -> CodeLM sees 256 codes, matching bpelm's context.

    uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_ifsq_2xflops_leanparams.py
"""
from pathlib import Path

K = 4
context_len = 1024
attn_window = 64
dq = 6
quant_type = "ifsq"
fsq_levels = 8

d_model = 96
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
