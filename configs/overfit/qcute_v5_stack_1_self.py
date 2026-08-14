"""v5_concat, softmax quant_type, single-level K=1, n_layers=1, gumbel disabled, STE on,
overfit10k testbed. Variant of qcute_v5_concat_overfit10k_k4single.py with Ks=(1,), n_layers=1.

uv run python -m qcute.qcute_v5_stack --config configs/overfit/qcute_v5_stack_1.py
"""
from pathlib import Path

Ks = (1,)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (32,)
cross_track_source = "decode"
decode_code_ste = True
share_level_weights = False
use_gumbel_noise = False
# gumbel_tau = 1.0

decode_separate_stage0 = False
# decode_self_only_aux = True
# decode_ntp_weight = 0.0
decode_ntp_weight = 1.0
# decode_self_only_weight = 1.0
data = Path("datasets/enwik8_1M.gz")
n_bytes = 1000
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
