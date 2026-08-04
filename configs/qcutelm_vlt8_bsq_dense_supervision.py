"""qcute.qcutelm_vlt8 config: "dense supervision" — all four loss edges
active simultaneously (main NTP always-on for both phases, plus all
three optional cross-component edges):
  code_match_weight=1.0    (existing: Encode's true code -> CodeLM's target)
  aux_recon_weight=1.0     (new: Encode's true code -> Decode, short direct
                             path for code_pre, block-local reconstruction)
  encode_match_weight=1.0  (new: CodeLM's own prediction -> Encode, mutual
                             consistency, directly trains the codelm-
                             prediction <-> true-code gap that tokenizer/
                             detokenizer-free free-rolling depends on)

Direct comparison point against qcutelm_vlt8_bsq.py (only code_match_weight
active) and qcutelm_vlt8_bsq_ntp_only.py (code_match_weight=0, no cross-
component supervision at all) — same bsq/dq=13/d_model=96/lm_d_model=256/
attn_window=80 architecture throughout, isolating loss composition as the
only variable across all three runs.

    uv run python -m qcute.qcutelm_vlt8 --config configs/qcutelm_vlt8_bsq_dense_supervision.py
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
aux_recon_weight = 1.0
encode_match_weight = 1.0
shared_tokenizer_phases = True  # pinned explicitly — isolates loss composition as the only variable
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
