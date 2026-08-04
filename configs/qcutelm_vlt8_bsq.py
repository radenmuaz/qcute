"""qcute.qcutelm_vlt8 config: BSQ quantizer, same architecture scale as
qcutelm_vlt7_bsq.py (dq=13, d_model=96/n_layers=2 tokenizer, lm_d_model=
256/n_layers=3 codelm, context_len=1024) — the ONLY difference is
attn_window=80 instead of 64: 80 = 16*(K+1), a clean multiple of the
block period (K+1=5), so every attention chunk covers exactly 16 whole
blocks (code slot included) instead of splitting a block's bytes from
its own code slot across a chunk boundary (see qcutelm_vlt8.py's module
docstring for the misalignment this fixes). Direct comparison point
against qcutelm_vlt7_bsq.py's result, isolating window-alignment as the
only variable.

    uv run python -m qcute.qcutelm_vlt8 --config configs/qcutelm_vlt8_bsq.py
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
attn_window = 80  # 16*(K+1) — block-aligned, see module docstring

lm_d_model = 256
lm_n_heads = 4
lm_n_layers = 3
lm_mlp_mult = 4

code_match_weight = 1.0
shared_tokenizer_phases = True  # pinned explicitly for reproducibility — this run started before the
                                 # untied-by-default change; qcutelm_vlt8_bsq_untied.py is the new-default
                                 # comparison point

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
