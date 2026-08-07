"""qcute.qcutelm_vlt6 config: RoPE (not NoPE) + no zero-KV sink for both
the tokenizer and CodeLM — matches qcute.bpelm's CausalSelfAttention
exactly architecturally (see qcutelm_vlt6.py's ZeroKVCausalSelfAttention
docstring: use_rope=True + use_zero_kv=False reproduces bpelm's plain
is_causal=True SDPA + RoPE, no sink). main_ntp_weight=1.0 only (aux_recon
and code_match both 0) — single-loss, matching the original design's
objective shape too, isolating the positional/attention-scheme change as
the sole variable vs qcutelm_vlt6_ifsq_vs_bpelm.py (NoPE+zero-KV).

Same architecture scale as qcutelm_vlt6_ifsq_2xflops_leanparams.py
otherwise — d_model=96/n_layers=2 (tokenizer), lm_d_model=256/n_layers=3
(codelm), context_len=1024 (256 codes, matching bpelm's context).
attn_window=-1 (full causal, no windowing) — bpelm itself has none.

    uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_rope_bpelm_parity.py
"""
from pathlib import Path

K = 4
context_len = 1024
attn_window = 64  # now uses the O(T*window) chunked no-sink path (_forward_chunked_no_sink, verified
                   # bit-exact against the dense no-sink path) — much faster than either the O(T^2)
                   # dense-windowed path or true full-causal (context_len=1024) at some cost to true
                   # bpelm parity (bpelm itself has no windowing at all).
dq = 5  # 8^5=32768 codespace — closer to bpelm_8192's 8192-token vocab than dq=6's 262144 (32x off),
        # though still 4x over (dq=4 would give 4096, 2x under — dq=5 was the explicit choice here)
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

use_rope = True
use_zero_kv = False

main_ntp_weight = 1.0
aux_recon_weight = 0.0
code_match_weight = 0.0

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
