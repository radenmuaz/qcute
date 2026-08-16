"""v5_concat, softmax quant_type, two-level K=(1,1), n_layers=1, gumbel disabled, STE on,
overfit10k testbed. Clone of qcute_v5_concat_1.py with decode_self_only_weight down to 0.1 --
tests whether the self-only auxiliary loss (which trains cleanly, see docs/status.md) is
crowding out cond_full's shared decode_lms[0] weights when both terms sum into the same step's
gradient (decode_self_only_dropout is off here, so both losses always backprop together).

uv run python -m qcute.qcute_v5_concat_slow --config configs/overfit/qcute_v5_concat_selfw0.1.py
"""
from pathlib import Path

Ks = (1,1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (32,32)
decode_pack_mode = "interleave"
decode_chunked = True
cross_track_source = "decode"
decode_code_ste = True
share_level_weights = False
use_gumbel_noise = False
gumbel_tau = 1.0

decode_self_only_aux = True
decode_self_only_weight = 0.1

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
eval_batches = 5

qual_gen_bytes = 32
qual_prompt_bytes = 16
