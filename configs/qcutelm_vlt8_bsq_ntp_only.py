"""qcute.qcutelm_vlt8 config: NTP-only baseline — code_match_weight=0, so
codelm gets ZERO direct supervision of its own (no target for its
prediction at all). The only active losses are loss_nocode (Pass 1) and
loss_decode (Pass 2), both plain byte-level NTP — exactly "how a regular
LM trains its last layer," with every earlier component (codelm, the
encoder's code_pre) getting gradient purely via backprop through the
final output loss, no auxiliary/intermediate targets anywhere. Tests
whether codelm can learn something useful purely indirectly, the way any
ordinary hidden layer in a deep network does, without an explicit
code-matching signal telling it what to predict.

Otherwise identical to qcutelm_vlt8_bsq.py — same bsq/dq=13/d_model=96/
lm_d_model=256/attn_window=80 architecture.

    uv run python -m qcute.qcutelm_vlt8 --config configs/qcutelm_vlt8_bsq_ntp_only.py
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

code_match_weight = 0.0  # the whole point of this config
shared_tokenizer_phases = True  # pinned explicitly — isolates code_match_weight as the only variable
                                 # against qcutelm_vlt8_bsq.py (which also used True); the untied-by-
                                 # default change happened after this config was written

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
