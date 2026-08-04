"""qcute.qcutelm_vlt7 config: combines both tokenizer-tier ablations —
trainable_slot_embed=True (learned "no-code" marker instead of zero) AND
shared_tokenizer_phases=False (Pass 1/Pass 2 get independent byte_emb/
blocks/ln_f/head weights instead of aliasing each other, same
architecture — mirrors qcutelm_vlt6's shared_encoder_decoder=False).
Otherwise identical to qcutelm_vlt7_bsq.py — same bsq/dq=13/d_model=96/
lm_d_model=256/attn_window=64 architecture — so results are directly
comparable across all three qcutelm_vlt7_bsq* runs (baseline, trainable-
slot-only, and this one).

    uv run python -m qcute.qcutelm_vlt7 --config configs/qcutelm_vlt7_bsq_trainable_slot_untied.py
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

code_match_weight = 1.0
trainable_slot_embed = True
shared_tokenizer_phases = False

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
