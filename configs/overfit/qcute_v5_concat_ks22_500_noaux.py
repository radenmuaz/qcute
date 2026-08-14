"""v5_concat, Ks=(2,2), n_layers=1, 1k-byte testbed, 500 steps -- decode_self_only_aux OFF.
Redo of qcute_v5_concat_ks22_500.py (which FAILED: neither cond_full nor cond_self matched
ground truth, cond_self badly garbled) with the aux curriculum loss disabled, to check whether it
was interfering with convergence at this Ks.

uv run python -m qcute.qcute_v5_concat --config configs/overfit/qcute_v5_concat_ks22_500_noaux.py
"""
from pathlib import Path

Ks = (2, 2)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (32, 32)
decode_pack_mode = "interleave"
decode_chunked = True
cross_track_source = "decode"
decode_code_ste = True
share_level_weights = False
use_gumbel_noise = False
gumbel_tau = 1.0

decode_self_only_aux = False

data = Path("datasets/enwik8_1M.gz")
n_bytes = 1000
val_frac = 0.1

steps = 500
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 32
qual_prompt_bytes = 16
