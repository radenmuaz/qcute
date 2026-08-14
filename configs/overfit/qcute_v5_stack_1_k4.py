"""Diagnostic: same as qcute_v5_stack_1.py but Ks=(4,1) instead of (1,1) -- tests whether
level0's self-cond collapse is a K=1-specific artifact (self-code is then a lossy *copy* of
information the cross-attention query already has in full from h_list[0], since there's no block
boundary for the code to usefully bridge) rather than a general architectural defect.

    uv run python -m qcute.qcute_v5_stack --config configs/overfit/qcute_v5_stack_1_k4.py
"""
from pathlib import Path

Ks = (4, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (32, 32)
cross_track_source = "decode"
decode_code_ste = True
share_level_weights = False
use_gumbel_noise = False

decode_separate_stage0 = False
decode_self_only_aux = True
decode_ntp_weight = 1.0
decode_self_only_weight = 1.0

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 1000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 10

qual_gen_bytes = 32
qual_prompt_bytes = 16
