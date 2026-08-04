"""qcute.qcutelm_vlt8 config: shared_tokenizer_phases=False, promoted from
"one ablation among several" to the theoretically-motivated default.

Session finding: encode and decode were never actually symmetric.
Decode is a genuine CONSUMER of codelm's output (every block conditioned
on a predicted code). Encode is a pure PRODUCER (computes z_hat from raw
bytes alone, never consumes codelm's output in return) — a one-way
pipeline (encode -> codelm -> decode), not a symmetric pair. True
symmetry would require encode to also consume codelm's forecast as
conditioning, which needs a block-by-block AR handshake between encoder
and codelm (encode pauses every K bytes, gets a code from codelm, resumes)
— expensive (loses full-sequence parallel training) and circular for no
clear benefit. Given they're actually different functions (unconditional
vs. conditional LM), sharing weights between them was asking one set of
weights to serve two incompatible roles, not a genuine architectural
elegance. This config drops that assumption.

Otherwise identical to qcutelm_vlt8_bsq.py — same bsq/dq=13/d_model=96/
lm_d_model=256/attn_window=80 architecture, direct comparison point.

    uv run python -m qcute.qcutelm_vlt8 --config configs/qcutelm_vlt8_bsq_untied.py
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
attn_window = 80

lm_d_model = 256
lm_n_heads = 4
lm_n_layers = 3
lm_mlp_mult = 4

code_match_weight = 1.0
shared_tokenizer_phases = False  # the point of this config

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
