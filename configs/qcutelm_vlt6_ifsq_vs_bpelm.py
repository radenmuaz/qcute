"""qcute.qcutelm_vlt6 config: matched against bpelm_8192, not bytelm_xs.

bpelm's context=256 means 256 BPE TOKENS (~4 bytes/token average — see
bpelm_8192.py's "matched-bandwidth" docstring), i.e. ~1024 raw bytes, not
256. qcutelm_vlt6's codes are structurally bpelm's tokens' analogue (each
compresses K=4 raw bytes), not bytelm's raw-byte analogue — so the apt
match is CodeLM's own context (in code-units) = bpelm's 256, not
context_len/K derived from bytelm's 256-byte span (that was
qcutelm_vlt6_ifsq_cheap_tokenizer.py, matched against bytelm instead).

context_len=1024 (K=4) -> CodeLM sees exactly 256 codes, matching bpelm's
256-token context directly. Same cheap-tokenizer/big-CodeLM split as
qcutelm_vlt6_ifsq_cheap_tokenizer.py (0.133M tokenizer, 3.184M CodeLM —
close to bpelm's own ~3.7M).

    uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_ifsq_vs_bpelm.py
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
lm_n_layers = 4
lm_mlp_mult = 4

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000  # 4x bytelm_xs_mtp4.py's 2000-step budget — enough room to see if/where this overfits, like bytelm did
batch_size = 16
lr_peak = 6e-4
weight_decay = 1e-5
warmup_steps = 500
cosine_decay = False  # bytelm_xs_mtp4.py doesn't use cosine decay either
constant_steps = 100
eval_every = 100
eval_batches = 20

gen_every = 200
gen_prompt_len = 64
gen_new_bytes = 64
