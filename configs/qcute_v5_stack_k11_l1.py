"""v5_stack, softmax quant_type, two-level K=(1,1), n_layers=1, gumbel disabled, STE on,
overfit10k testbed. All NTP loss weights on (decode_ntp_weight=1.0, the default) plus
decode_self_only_aux -- the standard level=2 config, run only after
qcute_v5_stack_k1_l1.py's level=1 baseline passes (see docs/status.md).

uv run python -m qcute.qcute_v5_stack --config configs/qcute_v5_stack_k11_l1.py
"""
from pathlib import Path

Ks = (1,1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (32,32)
cross_track_source = "decode"
decode_code_ste = True
share_level_weights = False
use_gumbel_noise = False

decode_separate_stage0 = False
decode_self_only_aux = True
decode_self_only_weight = 1.0

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 1000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 50
eval_every = 50
eval_batches = 10

qual_gen_bytes = 32
qual_prompt_bytes = 16
